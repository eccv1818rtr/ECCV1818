#!/usr/bin/env python3


import numpy as np
from typing import Dict, Any, Tuple


def seg_len(segments: np.ndarray, type: str = 'union') -> float:
    """
    get accumulated length of all line segments
    union: the intersection area is calculated only once
    sum: the intersection area is calculated several times
    Parameters
    ----------
    segments : shape (N, 2)
        each row is a segment with (start, end)

    Returns
    -------
    len : float
        total length of the union set of the segments
    """

    if type != 'union':
        return np.sum(segments[:, 1] - segments[:, 0]).item()

    segments_to_sum = []
    # sort by start coord
    segments = sorted(segments.tolist(), key=lambda x: x[0])
    for segment in segments:
        if len(segments_to_sum) == 0:
            segments_to_sum.append(segment)
            continue

        last_segment = segments_to_sum[-1]
        # if no overlap, append then merge
        if last_segment[1] < segment[0]:
            segments_to_sum.append(segment)
        else:
            union_segment = [min(last_segment[0], segment[0]), max(last_segment[1], segment[1])]
            segments_to_sum[-1] = union_segment

    segments_to_sum = np.array(segments_to_sum, dtype=np.float32)
    return np.sum(segments_to_sum[:, 1] - segments_to_sum[:, 0]).item()


def calc_inter(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate intersection boxes and areas of each pred and gt box
    Parameters
    ----------
    pred_boxes : shape (N, 4)
    gt_boxes : shape (M, 4)
    box format top-left and bottom-right coords (x1, y1, x2, y2)

    Returns
    -------
    inter_boxes : numpy.ndarray, shape (N, M, 4)
        intersection boxes of each pred and gt box
    inter_areas : numpy.ndarray, shape (N, M)
        intersection areas of each pred and gt box
    """
    lt = np.maximum(pred_boxes[:, None, :2], gt_boxes[:, :2])
    rb = np.minimum(pred_boxes[:, None, 2:], gt_boxes[:, 2:])
    wh = np.maximum(rb - lt, 0)
    inter_boxes = np.concatenate((lt, rb), axis=2)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]
    return inter_boxes, inter_areas


def precision_recall(pred_boxes: np.ndarray, gt_boxes: np.ndarray):
    """
    Segment level Precision/Recall evaluation for one video pair vta result
    pred_boxes shape(N, 4) indicates N predicted copied segments
    gt_boxes shape(M, 4) indicates M ground-truth labelled copied segments
    Parameters
    ----------
    pred_boxes : shape (N, 4)
    gt_boxes : shape (M, 4)

    Returns
    -------
    precision : float
    recall : float
    """

    # abnormal assigned values for denominator = 0
    if len(pred_boxes) > 0 and len(gt_boxes) == 0:
        return {"precision": 0, "recall": 1}

    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        return {"precision": 1, "recall": 0}

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"precision": 1, "recall": 1}

    # intersection area calculation
    inter_boxes, inter_areas = calc_inter(pred_boxes, gt_boxes)

    sum_tp_w, sum_p_w, sum_tp_h, sum_p_h = 0, 0, 0, 0
    for pred_ind, inter_per_pred in enumerate(inter_areas):
        # for each pred-box, find the gt-boxes whose iou is > 0 with it
        pos_gt_inds = np.where(inter_per_pred > 0)
        if len(pos_gt_inds[0]) > 0:

            # union of all pred box along each side
            # tp: true positive
            sum_tp_w += seg_len(np.squeeze(inter_boxes[pred_ind, pos_gt_inds, :][:, :, [0, 2]], axis=0))

            sum_tp_h += seg_len(np.squeeze(inter_boxes[pred_ind, pos_gt_inds, :][:, :, [1, 3]], axis=0))

    sum_p_w = seg_len(pred_boxes[:, [0, 2]], type='sum')
    sum_p_h = seg_len(pred_boxes[:, [1, 3]], type='sum')
    precision_w = sum_tp_w / (sum_p_w + 1e-6)
    precision_h = sum_tp_h / (sum_p_h + 1e-6)

    sum_tp_w, sum_p_w, sum_tp_h, sum_p_h = 0, 0, 0, 0
    for gt_ind, inter_per_gt in enumerate(inter_areas.T):
        # for each gt-box, find the pred-boxes whose iou is > 0 with it
        pos_pred_inds = np.where(inter_per_gt > 0)
        if len(pos_pred_inds[0]) > 0:

            # union of all pred box along each side
            # tp: true positive
            sum_tp_w += seg_len(np.squeeze(inter_boxes[pos_pred_inds, gt_ind, :][:, :, [0, 2]], axis=0))

            sum_tp_h += seg_len(np.squeeze(inter_boxes[pos_pred_inds, gt_ind, :][:, :, [1, 3]], axis=0))

    sum_p_w = seg_len(gt_boxes[:, [0, 2]], type='sum')
    sum_p_h = seg_len(gt_boxes[:, [1, 3]], type='sum')
    recall_w = sum_tp_w / (sum_p_w + 1e-6)
    recall_h = sum_tp_h / (sum_p_h + 1e-6)

    return {"precision": precision_h * precision_w, "recall": recall_h * recall_w}


def evaluate_overall(result_dict: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
    """
    This metric indicates the overall performance on all the video pairs.
    Parameters
    ----------
    result_dict: segment level Precision/Recall result of all the video pairs
    ratio: nums of positive samples / nums of negative samples

    Returns
    -------
    recall, precision
    """
    # The following metric directly filter out the result with abnormal assigned values
    # if len(pred_boxes) == 0, precision is 1 but it will not contribute to final precision metric.
    # max value of regular calculation is always != 1 since x / (x + 1e-6) < 1
    precision_list = [result_dict[i]['precision'] for i in result_dict if not (result_dict[i]['precision'] == 1)]
    # if len(gt_boxes) == 0, recall is 1 but it will not contribute to final recall metric.
    recall_list = [result_dict[i]['recall'] for i in result_dict if not (result_dict[i]['recall'] == 1)]
    # r, p = sum(recall_list) / len(recall_list), sum(precision_list) / len(precision_list)
    r = sum(recall_list) / len(recall_list) if len(recall_list) > 0 else 0
    p = sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0

    return r, p
