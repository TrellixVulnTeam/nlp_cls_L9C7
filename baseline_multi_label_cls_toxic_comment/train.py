# -*-coding:utf-8-*-
import os
import sys
import json
import torch
import logging
import numpy as np

cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)

from tqdm import tqdm
from tqdm import trange
from sklearn.metrics import roc_curve, auc
from baseline_multi_label_cls_toxic_comment.config import parse_args
from baseline_multi_label_cls_toxic_comment.config import init_seed
from baseline_multi_label_cls_toxic_comment.data_util import MultiLabelTextProcessor
from baseline_multi_label_cls_toxic_comment.data_util import convert_examples_to_features
from baseline_multi_label_cls_toxic_comment.data_util import convert_features_to_dataset
from baseline_multi_label_cls_toxic_comment.data_util import init_dataloader
from baseline_multi_label_cls_toxic_comment.evaluate import accuracy_thresh
from baseline_multi_label_cls_toxic_comment.model import Bert_CLS

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def init_bertadam_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1, t_total=args.t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    return model, optimizer


def train(args, model, train_dataloader, dev_dataloader):
    """ Train the model """

    model, optimizer = init_bertadam_optimizer(args, model)

    global_step = 0
    global_accu = 0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=-1 not in [-1, 0])
    for epoch in train_iterator:
        epoch_loss, epoch_step = 0, 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=-1 not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            device = torch.device("cuda")
            batch = tuple(t.to(device) for t in batch)

            inputs = {'input_ids':      batch[0].long(),
                      'token_type_ids': None,
                      'attention_mask': None,
                      'labels':         batch[3].long()}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()

            epoch_loss += loss.item()
            epoch_step += 1
            global_step += 1

            # AdamW Optim
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if global_step % args.test_steps == 0:
                result = eval(args, model, dev_dataloader)
                logger.info("step: {}, eval_loss: {}, eval_accu: {}".format(
                    global_step, result["eval_loss"], result["eval_acc"]))

                if result['eval_acc'] > global_accu:
                    global_accu = result['eval_acc']
                    result_path = os.path.join(args.checkpoint, f"result_step{global_step}.json")
                    model_path  = os.path.join(args.checkpoint, f"model.bin")
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), model_path)

                    with open(result_path, "w", encoding="utf-8") as writer:
                        json.dump(result, writer, ensure_ascii=False, indent=4)

        train_loss = epoch_loss / epoch_step
        result = eval(args, model, dev_dataloader)
        logger.info("epoch: {}, train_loss: {}, eval_loss: {}, eval_accu: {}".format(
            epoch, train_loss, result["eval_loss"], result["eval_acc"]))

        if result['eval_acc'] > global_accu:
            global_accu = result['eval_acc']
            result_path = os.path.join(args.checkpoint, f"result_step{global_step}.json")
            model_path  = os.path.join(args.checkpoint, f"model.bin")
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), model_path)

            with open(result_path, "w", encoding="utf-8") as writer:
                json.dump(result, writer, ensure_ascii=False, indent=4)


def eval(args, model, dev_dataloader):
    preds = None
    out_label_ids = None
    all_logits = None
    all_labels = None
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(dev_dataloader, desc="Evaluating", disable=-1 not in [-1, 0]):
        model.eval()
        device = torch.device("cuda")
        batch = tuple(t.to(device) for t in batch)

        inputs = {'input_ids':      batch[0].long(),
                  'token_type_ids': batch[1].long(),
                  'attention_mask': batch[2].long(),
                  'labels':         batch[3].long()}
        input_ids = batch[0]
        label_ids = batch[3]
        with torch.no_grad():
            outputs = model(**inputs)

        tmp_eval_loss , logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss  = tmp_eval_loss.mean()

        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(args.num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    result = {
        'eval_loss': eval_loss,
        'eval_acc': eval_accuracy,
        'roc_auc': roc_auc
    }
    return result


if __name__ == "__main__":
    args = parse_args()
    init_seed(args)
    os.makedirs(args.checkpoint, exist_ok=True)

    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.per_train_bsz * args.n_gpu
    args.dev_batch_size   = args.per_dev_bsz   * args.n_gpu

    max_seq_length = 512
    bert_path = os.path.join(par_dir, "pretrained_model/bert-base-uncased/")
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    data_dir = os.path.join(par_dir, "dataset/toxic_comment")
    predict_processor = MultiLabelTextProcessor(data_dir)
    label_list = predict_processor.get_labels()
    train_examples = predict_processor.get_train_examples(data_dir)
    dev_examples   = predict_processor.get_dev_examples(data_dir)
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
    dev_features   = convert_examples_to_features(dev_examples,   label_list, max_seq_length, tokenizer)
    train_dataset  = convert_features_to_dataset(train_features)
    dev_dataset    = convert_features_to_dataset(dev_features)
    train_loader = init_dataloader(train_dataset, data_type="train", batch_size=args.train_batch_size)
    dev_loader   = init_dataloader(dev_dataset,   data_type="dev",   batch_size=args.dev_batch_size)

    args.num_labels = len(label_list)

    args.t_total = len(train_loader) * args.num_train_epochs
    args.warmup_steps = int(args.t_total * args.warmup_rate)

    args_path = os.path.join(args.checkpoint, "args.json")
    with open(args_path, "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, ensure_ascii=False, indent=4)

    logger.info("init model")
    config_path = os.path.join(args.bert_path, "bert_config.json")
    from pytorch_pretrained_bert.modeling import BertConfig
    bert_config = BertConfig.from_json_file(config_path)
    model = Bert_CLS.from_pretrained(args.bert_path, num_labels=6).cuda()
    train(args, model, train_loader, dev_loader)
