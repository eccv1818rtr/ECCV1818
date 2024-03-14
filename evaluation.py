import argparse
from pathlib import Path

import pandas as pd
import os
import json
import numpy as np

from metric.eval import *
from multiprocessing import Pool
from loguru import logger


def process_conf(eval):
    key, gt, pred_box = eval["name"], eval['gt'], np.array(eval['pred'])
    try:
        return {"name": key, "gt": gt, "pred": pred_box[pred_box[:, -1] > conf][:, :4].astype(int).tolist()}
    except:
        return {"name": key, "gt": gt, "pred": []}


def eval(input_dict):
    gt_box = np.array(input_dict["gt"])
    pred_box = np.array(input_dict["pred"])
    result_dict = precision_recall(pred_box, gt_box)
    result_dict["name"] = input_dict["name"]
    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", type=str, default=None, help="gt label file")
    parser.add_argument("--pred-file", type=str, default=None, help="result dir of video segment prediction")
    parser.add_argument("--test-file", type=str, default=None, help="test pair list of query and reference videos")
    parser.add_argument("--pool-size", type=int, default=16, help="multiprocess pool size of evaluation")
    parser.add_argument("--conf", type=float, default=0.1, help="input with conf")

    args = parser.parse_args()

    logger.remove(handler_id=None)
    logger.add(Path(args.pred_file).parent / f"results.log")
    logger.info(f"start loading...")

    df = pd.read_csv(args.test_file)
    split_pairs = set([f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)])

    gt = json.load(open(args.anno_file))
    key_list = [key for key in gt]

    process_pool = Pool(args.pool_size)

    pred_dict = json.load(open(args.pred_file))
    eval_list = []
    for key in split_pairs:
        if key in gt:
            if key in pred_dict:
                eval_list += [{"name": key, "gt": gt[key], "pred": pred_dict[key]}]
            else:
                eval_list += [{"name": key, "gt": gt[key], "pred": []}]
        else:
            if key in pred_dict:
                eval_list += [{"name": key, "gt": [], "pred": pred_dict[key]}]
            else:
                eval_list += [{"name": key, "gt": [], "pred": []}]

    if args.conf != 0:
        conf = args.conf
        process_pool = Pool(args.pool_size)
        eval_list = process_pool.map(process_conf, eval_list)

    process_pool = Pool(args.pool_size)
    result_list = process_pool.map(eval, eval_list)
    result_dict = {i['name']: i for i in result_list}
    r, p = evaluate_overall(result_dict)

    logger.info(f"result file: {args.pred_file}, "
                f"data cnt: {len(result_list)}, "
                f"micro-Recall: {r:.2%}, "
                f"micro-Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p + 1e-6):.2%}")

    print(f"data cnt: {len(result_list)}, "
          f"micro-Recall: {r:.2%}, "
          f"micro-Precision: {p:.2%}, "
          f"F1: {2 * r * p / (r + p + 1e-6):.2%}")

    ratio = 1
    result_list = list(result_dict.keys())
    # obtain video-level false reject (missing) samples and false alarm samples nums
    fr_list = [i for i in result_dict if result_dict[i]['precision'] == 1 and result_dict[i]['recall'] == 0]
    fa_list = [i for i in result_dict if result_dict[i]['precision'] == 0 and result_dict[i]['recall'] == 1]

    # for key in fr_list:
    #     print("%s;%s" % (key, str(gt[key])))

    # total positive samples are len(result_list) * ratio / (1 + ratio)
    # total negative samples are len(result_list) * 1 / (1 + ratio)

    fp = len(fa_list)
    tp = len(result_list) * ratio / (1 + ratio) - len(fr_list)
    pp = fp + tp
    p = tp / pp
    r = tp / (len(result_list) * ratio / (1 + ratio))

    logger.info(f"Video Level Recall: {r:.2%}, "
                f"Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p + 1e-6):.2%}, ")

    print(f"Video Level Recall: {r:.2%}, "
          f"Precision: {p:.2%}, "
          f"F1: {2 * r * p / (r + p + 1e-6):.2%}, ")
