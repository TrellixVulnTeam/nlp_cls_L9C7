import os
import sys
import json
import torch
import pickle
cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset
from pytorch_pretrained_bert.tokenization import BertTokenizer


class Example:
    def __init__(self, qas_id, question, labels):
        self.qas_id = qas_id
        self.question = question
        self.labels = labels


class Input_Feature:
    def __init__(self, qas_id, input_ids, token_type_ids, input_mask, labels):
        self.qas_id = qas_id
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.input_mask = input_mask
        self.labels = labels


def convert_example(data_type):
    dataset_dir  = os.path.join(par_dir, "dataset/sent_med")
    data_path    = os.path.join(dataset_dir, f"{data_type}.json")
    example_path = os.path.join(dataset_dir, f"{data_type}_example.pkl")
    with open(data_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)

    examples = []
    for item in input_data:
        labels = [0] * 6
        if item["category_A"] == 1: labels[0] = 1
        if item["category_B"] == 1: labels[1] = 1
        if item["category_C"] == 1: labels[2] = 1
        if item["category_D"] == 1: labels[3] = 1
        if item["category_E"] == 1: labels[4] = 1
        if item["category_F"] == 1: labels[5] = 1

        example = Example(qas_id=item["ID"], question=item["Question_Sentence"], labels=labels)
        examples.append(example)

    with open(example_path, "wb") as writer:
        pickle.dump(examples, writer)


def convert_feature(data_type):
    bert_path = os.path.join(par_dir, "pretrained_model/bert-base-chinese/")
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    dataset_dir = os.path.join(par_dir, "dataset/sent_med")
    example_path = os.path.join(dataset_dir, f"{data_type}_example.pkl")
    feature_path = os.path.join(dataset_dir, f"{data_type}_feature.pkl")

    with open(example_path, "rb") as reader:
        examples = pickle.load(reader)

    max_seq_length = 128
    features = []
    for example in examples:
        question = example.question

        question_tokens = tokenizer.tokenize(question)
        question_tokens = question_tokens[:max_seq_length-2]

        input_tokens = ["[CLS]"] + question_tokens + ["[SEP]"]
        input_ids      = tokenizer.convert_tokens_to_ids(input_tokens)
        token_type_ids = [0] * len(input_tokens)
        input_mask     = [1] * len(input_tokens)

        while len(input_tokens) < max_seq_length:
            input_tokens.append("[PAD]")
            token_type_ids.append(0)
            input_mask.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        feature = Input_Feature(qas_id=example.qas_id,
                                input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                input_mask=input_mask,
                                labels=example.labels)
        features.append(feature)

    with open(feature_path, "wb") as writer:
        pickle.dump(features, writer)


def convert_dataset(data_type):
    dataset_dir = os.path.join(par_dir, "dataset/sent_med")
    feature_path = os.path.join(dataset_dir, f"{data_type}_feature.pkl")
    dataset_path = os.path.join(dataset_dir, f"{data_type}_dataset.pkl")
    with open(feature_path, "rb") as reader:
        features = pickle.load(reader)

    # Convert to Tensors and build dataset
    all_input_ids      = torch.tensor([feature.input_ids      for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_input_mask     = torch.tensor([feature.input_ids      for feature in features], dtype=torch.long)
    all_labels         = torch.tensor([feature.labels         for feature in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_input_mask, all_labels)

    with open(dataset_path, "wb") as writer:
        pickle.dump(dataset, writer)


def init_dataloader(data_type, batch_size):
    dataset_dir = os.path.join(par_dir, "dataset/sent_med")
    dataset_path = os.path.join(dataset_dir, f"{data_type}_dataset.pkl")
    with open(dataset_path, "rb") as reader:
        dataset = pickle.load(reader)

    sampler    = RandomSampler(dataset) if data_type == "train" else SequentialSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


if __name__ == "__main__":
    convert_example(data_type="train")
    convert_feature(data_type="train")
    convert_dataset(data_type="train")

    convert_example(data_type="dev")
    convert_feature(data_type="dev")
    convert_dataset(data_type="dev")
