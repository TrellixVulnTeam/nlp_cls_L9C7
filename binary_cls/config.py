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

    # learning_rate = 0.1   # sgd,  eval_accu:
    # learning_rate = 0.01  # adam, eval_accu: 0.786697247706422
    # learning_rate = 5e-3  # adam, eval_accu:
    learning_rate = 1e-3  # adam, eval_accu: 0.8004587155963303
    # learning_rate = 5e-4  # adam, eval_accu: 0.797018348623853200
    # learning_rate = 1e-4  # adam, eval_accu: 0.786697247706422
    print("learning_rate = ", learning_rate)
    test_steps    = 500
    adam_epsilon  = 1e-8
    per_train_bsz = 32
    per_dev_bsz   = 16
    dataset_type = "SST-2"  # SST-2 / CoLA

    # # Required parameters
    checkpoint = os.path.join(par_dir, f"checkpoint/binary_cls/{dataset_type}/cnn/01")
    parser.add_argument("--checkpoint",   type=str, default=checkpoint)
    parser.add_argument("--dataset_type", type=str, default=dataset_type)

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

    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--hidden_size",    type=int, default=100)
    parser.add_argument("--vocab_size",     type=int, default=0)

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