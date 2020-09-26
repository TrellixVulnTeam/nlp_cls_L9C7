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
from binary_cls.model_baseline.model import NGram_CLS
from torch.optim import SGD
from torch.optim import Adam

from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Example
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from nltk.tokenize import word_tokenize

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


def train(args, model, train_iterator, dev_iterator):
    """ Train the model """
    # model, optimizer = init_sgd_optimizer(args, model)
    model, optimizer = init_adam_optimizer(args, model)

    global_step = 0
    global_accu = 0
    for epoch in range(args.num_train_epochs):
        epoch_loss, epoch_step = 0, 0
        for batch in next(iter(train_iterator)):
            model.train()

            input_ids, input_len = batch.sent
            labels = batch.label

            batch_size  = input_ids.size(0)
            max_seq_len = input_ids.size(1)
            input_mask = torch.arange(max_seq_len).expand(batch_size, max_seq_len).cuda() < input_len.unsqueeze(1)

            inputs = {'input_ids':  input_ids.long().cuda(),
                      'input_mask': input_mask.long().cuda(),
                      'labels':     labels.long().cuda()}
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
                result = eval(args, model, dev_iterator)
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
        result = eval(args, model, dev_iterator)
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


def eval(args, model, dev_iterator):
    preds = None
    out_label_ids = None
    global_loss, global_step = 0, 0
    for batch in next(iter(dev_iterator)):
        model.eval()
        input_ids, input_len = batch.sent
        labels = batch.label

        batch_size  = input_ids.size(0)
        max_seq_len = input_ids.size(1)
        input_mask  = torch.arange(max_seq_len).expand(batch_size, max_seq_len).cuda() < input_len.unsqueeze(1)
        inputs = {'input_ids':  input_ids.long().cuda(),
                  'input_mask': input_mask.long().cuda(),
                  'labels':     labels.long().cuda()}

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


def build_iterator(args):

    def tokenize(text):
        words = word_tokenize(text)
        return words

    QAS_ID = Field(sequential=False, use_vocab=False)
    SENT   = Field(sequential=True,  tokenize=tokenize, batch_first=True, lower=True, fix_length=32, include_lengths=True)
    LABEL  = Field(sequential=False, use_vocab=False,   batch_first=True, include_lengths=False)
    # field = {'qas_id': ('qas_id', QAS_ID), 'sent': ('sent', SENT), 'label': ('label', LABEL)}
    field = {'sent': ('sent', SENT), 'label': ('label', LABEL)}
    # field = [(None, None), ('sent', SENT), ('label', LABEL)]

    dataset_path = os.path.join(par_dir, "dataset/SST-2/")
    train_data, dev_data, test_data = TabularDataset.splits(path=dataset_path,
                                                            train="train1.json",
                                                            validation="dev1.json",
                                                            test="test1.json",
                                                            format="json",
                                                            fields=field)

    # glove_name = 'fasttext.en.300d'
    # glove_name = 'fasttext.simple.300d'
    # glove_name = 'glove.42B.300d'
    # glove_name = 'glove.840B.300d'
    # glove_name = 'glove.twitter.27B.25d'
    # glove_name = 'glove.twitter.27B.50d'
    # glove_name = 'glove.twitter.27B.100d'
    # glove_name = 'glove.twitter.27B.200d'
    # glove_name = 'glove.6B.50d'
    # glove_name = 'glove.6B.100d'
    # glove_name = 'glove.6B.200d'
    # glove_name = 'glove.6B.300d'
    SENT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d')
    train_iterator = BucketIterator.splits([train_data],
                                           batch_size=args.train_batch_size,
                                           # sort_with_batch=True,
                                           # sort_key=lambda x:len(x.sent),
                                           device=torch.device("cuda"),
                                           shuffle=True)
    dev_iterator   = BucketIterator.splits([dev_data],
                                           batch_size=args.dev_batch_size,
                                           # sort_with_batch=True,
                                           # sort_key=lambda x: len(x.sent),
                                           device=torch.device("cuda"),
                                           shuffle=False)
    test_iterator  = BucketIterator.splits([test_data],
                                           batch_size=args.dev_batch_size,
                                           # sort_with_batch=True,
                                           # sort_key=lambda x: len(x.sent),
                                           device=torch.device("cuda"),
                                           shuffle=False)

    EMBEDDING_DIM = args.embedding_size
    UNK_IDX = SENT.vocab.stoi[SENT.unk_token]
    PAD_IDX = SENT.vocab.stoi[SENT.pad_token]
    SENT.vocab.vectors.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    SENT.vocab.vectors.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    embeddings = SENT.vocab.vectors
    # print("embeddings size = ", embeddings.size())
    return embeddings, train_iterator, dev_iterator, test_iterator


if __name__ == "__main__":
    args = parse_args()
    init_seed(args)
    os.makedirs(args.checkpoint, exist_ok=True)

    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.per_train_bsz * args.n_gpu
    args.dev_batch_size   = args.per_dev_bsz   * args.n_gpu

    embeddings, train_iterator, dev_iterator, test_iterator = build_iterator(args)
    args.vocab_size = embeddings.size(0)
    embeddings = embeddings.numpy()

    args_path = os.path.join(args.checkpoint, "args.json")
    with open(args_path, "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, ensure_ascii=False, indent=4)

    # model = NGram_CLS(embeddings=embeddings,
    #                   vocab_size=args.vocab_size,
    #                   embedding_size=args.embedding_size)
    model = TextCNN(embeddings=embeddings,
                    vocab_size=args.vocab_size,
                    embedding_size=args.embedding_size)
    model.cuda()
    train(args, model, train_iterator, dev_iterator)
