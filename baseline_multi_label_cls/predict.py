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
from baseline_multi_label_cls.config import parse_args
from baseline_multi_label_cls.config import init_seed
from baseline_multi_label_cls.data_util import init_dataloader
from baseline_multi_label_cls.model import Bert_CLS

from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_thresh(y_pred, thresh=0.5, sigmoid=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = y_pred > thresh
    return y_pred


def predict(args, model, test_dataloader):

    preds = None
    for batch in tqdm(test_dataloader, desc="Predicting", disable=-1 not in [-1, 0]):
        model.eval()
        device = torch.device("cuda")
        batch = tuple(t.to(device) for t in batch)

        inputs = {'input_ids':      batch[0].long(),
                  'token_type_ids': batch[1].long(),
                  'attention_mask': batch[2].long()}

        with torch.no_grad():
            logits = model(**inputs)

        if preds is None:
            preds = logits
        else:
            preds = torch.cat([preds, logits], dim=0)
    y_preds = predict_thresh(preds)
    y_preds = y_preds.cpu().detach().numpy().tolist()

    dataset_path = os.path.join(par_dir, "dataset/sent_med/dev.json")
    predict_path = os.path.join(par_dir, "dataset/sent_med/dev_pre.json")
    with open(dataset_path, "r", encoding="utf-8") as reader:
        input_data = json.load(reader)

    results = [{"qas_id": item["ID"], "labels": pred} for item, pred in zip(input_data, y_preds)]
    with open(predict_path, "w", encoding="utf-8") as writer:
        json.dump(results, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    args = parse_args()
    init_seed(args)
    os.makedirs(args.checkpoint, exist_ok=True)

    args.n_gpu = torch.cuda.device_count()
    args.test_batch_size = args.per_dev_bsz * args.n_gpu
    test_loader = init_dataloader(data_type="dev", batch_size=args.test_batch_size)

    logger.info("init model")
    config_path = os.path.join(args.bert_path, "bert_config.json")
    from pytorch_pretrained_bert.modeling import BertConfig
    bert_config = BertConfig.from_json_file(config_path)
    model = Bert_CLS(config=bert_config, num_labels=6).cuda()

    state_dict_path = os.path.join(par_dir, "checkpoint/baseline/bert_base/01/model.bin")
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    predict(args, model, test_loader)
