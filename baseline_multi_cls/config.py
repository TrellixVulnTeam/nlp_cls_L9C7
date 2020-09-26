import os
import sys
import torch
import random
import argparse
import numpy as np

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    bert_type = "base"
    bert_path = "/media/data1/luokunhao/pretrained_model/bert-base-chinese"
    learning_rate = 1e-5 if bert_type == "large" else 3e-5
    test_steps    = 500  if bert_type == "large" else 200
    adam_epsilon  = 1e-8
    per_train_bsz = 4    if bert_type == "large" else 8
    per_dev_bsz   = 8

    # # Required parameters
    checkpoint = os.path.join(par_dir, f"checkpoint/baseline_multi_cls/toutiao/bert_{bert_type}/01")
    parser.add_argument("--bert_type",  type=str, default=bert_type)
    parser.add_argument("--bert_path",  type=str, default=bert_path)
    parser.add_argument("--checkpoint", type=str, default=checkpoint)

    parser.add_argument("--max_grad_norm",    type=int, default=1)
    parser.add_argument("--n_gpu",            type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--seed",             type=int, default=5233)
    parser.add_argument("--learning_rate",    type=float, default=learning_rate)
    parser.add_argument("--test_steps",       type=int, default=test_steps)

    parser.add_argument("--train_batch_size", type=int, default=0)
    parser.add_argument("--dev_batch_size",   type=int, default=0)
    parser.add_argument("--per_train_bsz",    type=int, default=per_train_bsz)
    parser.add_argument("--per_dev_bsz",      type=int, default=per_dev_bsz)

    parser.add_argument("--adam_epsilon", type=float, default=adam_epsilon)
    parser.add_argument("--warmup_rate",  type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int,   default=0)
    parser.add_argument("--t_total",      type=int,   default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    return args


def init_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)