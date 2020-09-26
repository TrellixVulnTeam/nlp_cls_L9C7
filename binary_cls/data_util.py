import os
import sys
import json
import torch
cur_path = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_path)
par_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from binary_cls.glove_util import Word_Model
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset


class Feature:
    def __init__(self, input_ids, input_mask, labels):
        self.input_ids  = input_ids
        self.input_mask = input_mask
        self.labels     = labels


def convert_feature(data_type):
    data_path = os.path.join(par_dir, f"dataset/SST-2/{data_type}.json")
    with open(data_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)

    word_util = Word_Model()
    features = []
    max_seq_length = 32
    for item in tqdm(input_data):
        sent = item["sent"]
        words = word_tokenize(sent)
        words = words[:max_seq_length]
        input_mask = [1] * len(words)
        while len(words) < max_seq_length:
            words.append("PAD")
            input_mask.append(0)
        input_ids = [word_util.word2index(word) for word in words]
        feature = Feature(input_ids=input_ids, input_mask=input_mask, labels=item["label"])
        features.append(feature)
    return features


def convert_dataset(features):
    # Convert to Tensors and build dataset
    all_input_ids  = torch.tensor([feature.input_ids  for feature in features], dtype=torch.long)
    all_input_mask = torch.tensor([feature.input_mask for feature in features], dtype=torch.long)
    all_labels     = torch.tensor([feature.labels     for feature in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_labels)
    return dataset


def init_dataloader(dataset, data_type, batch_size):
    sampler    = RandomSampler(dataset) if data_type == "train" else SequentialSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
