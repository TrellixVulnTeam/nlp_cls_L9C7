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
from binary_cls.config import parse_args
from binary_cls.config import init_seed
from binary_cls.glove_util import Word_Model
from binary_cls.data_util import convert_feature
from binary_cls.data_util import convert_dataset
from binary_cls.data_util import init_dataloader
from binary_cls.model_cnn.textcnn import TextCNN
from binary_cls.model_rnn.textrnn import RNN_CLS
from binary_cls.model_rnn.textrnn import LSTM_CLS
from binary_cls.model_rnn.textrnn import GRU_CLS
from binary_cls.model_rnn.textrcnn import RCNN_CLS
from binary_cls.model_rnn.textrnn_attn import RNN_Attn_CLS
from torch.optim import SGD
from torch.optim import Adam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def init_sgd_optimizer(args, model):
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    return model, optimizer


def init_adam_optimizer(args, model):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    if args.n_gpu > 1:
        gpu_ids = list(range(args.n_gpu))
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    return model, optimizer


def train(args, model, train_dataloader, dev_dataloader):
    """ Train the model """
    # model, optimizer = init_sgd_optimizer(args, model)
    model, optimizer = init_adam_optimizer(args, model)

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

            inputs = {'input_ids':  batch[0].long(),
                      'input_mask': batch[1].long(),
                      'labels':     batch[2].long()}
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
    global_loss, global_step = 0, 0
    for batch in tqdm(dev_dataloader, desc="Evaluating", disable=-1 not in [-1, 0]):
        model.eval()
        device = torch.device("cuda")
        batch = tuple(t.to(device) for t in batch)

        inputs = {'input_ids':  batch[0].long(),
                  'input_mask': batch[1].long(),
                  'labels':     batch[2].long()}
        with torch.no_grad():
            outputs = model(**inputs)

        loss, logits = outputs[:2]
        if args.n_gpu > 1:
            loss = loss.mean()

        global_loss += loss.item()
        global_step += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=1)
    eval_acc = simple_accuracy(preds, out_label_ids)
    eval_loss = global_loss / global_step
    result = {"eval_acc": eval_acc, "eval_loss": eval_loss}
    return result


if __name__ == "__main__":
    args = parse_args()
    init_seed(args)
    os.makedirs(args.checkpoint, exist_ok=True)

    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.per_train_bsz * args.n_gpu
    args.dev_batch_size   = args.per_dev_bsz   * args.n_gpu

    train_features = convert_feature(data_type="train")
    dev_features   = convert_feature(data_type="dev")
    train_dataset  = convert_dataset(train_features)
    dev_dataset    = convert_dataset(dev_features)
    train_loader   = init_dataloader(train_dataset, data_type="train", batch_size=args.train_batch_size)
    dev_loader     = init_dataloader(dev_dataset,   data_type="dev",   batch_size=args.dev_batch_size)

    args.t_total = len(train_loader) * args.num_train_epochs
    args.warmup_steps = int(args.t_total * args.warmup_rate)

    args_path = os.path.join(args.checkpoint, "args.json")
    with open(args_path, "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, ensure_ascii=False, indent=4)

    logger.info("init model")
    word_model = Word_Model()
    args.vocab_size = word_model.get_vocab_size()
    embeddings = word_model.get_word_vecs()

    # eval_accu:
    # model = TextCNN(embeddings=embeddings,
    #                 vocab_size=args.vocab_size,
    #                 embedding_size=args.embedding_size)

    # model = RNN_CLS(embeddings=embeddings,
    #                 vocab_size=args.vocab_size,
    #                 embedding_size=args.embedding_size,
    #                 hidden_size=args.hidden_size,
    #                 num_labels=2)

    # model = LSTM_CLS(embeddings=embeddings,
    #                  vocab_size=args.vocab_size,
    #                  embedding_size=args.embedding_size,
    #                  hidden_size=args.hidden_size,
    #                  num_labels=2)

    # model = GRU_CLS(embeddings=embeddings,
    #                 vocab_size=args.vocab_size,
    #                 embedding_size=args.embedding_size,
    #                 hidden_size=args.hidden_size,
    #                 num_labels=2)

    # model = RCNN_CLS(embeddings=embeddings,
    #                  vocab_size=args.vocab_size,
    #                  embedding_size=args.embedding_size,
    #                  hidden_size=args.hidden_size,
    #                  num_labels=2)

    model = RNN_Attn_CLS(embeddings=embeddings,
                         vocab_size=args.vocab_size,
                         embedding_size=args.embedding_size,
                         hidden_size=args.hidden_size,
                         num_labels=2)

    model.cuda()
    train(args, model, train_loader, dev_loader)
