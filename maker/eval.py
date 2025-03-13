import argparse
import json

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from Levenshtein import ratio

from dataset import generate_datasets

# Extend coco class to accept dict
class COCO_from_dict(COCO):
    def __init__(self, annotation_dict):
        super(COCO_from_dict, self).__init__()
        self.dataset = annotation_dict
        self.createIndex()



# COCO eval and the following mAP, mR, and mP calculations return measurements in multiple dimensions due to different settings

# T = eval['counts'][0] # Number of iou thresholds (n in formula)
# R = eval['counts'][1] # Number of recall thresholds
# K = eval['counts'][2] # Number of classes (N in formula)
# A = eval['counts'][3] # Number of area range thresholds
# M = eval['counts'][4] # Number of max detection thresholds

# eval['recall'] has a shape [T, K, A, M]
# eval['precision'] has a shape [T, K, R, A, M]

def do_cocoeval(cocoGt, cocoDt, params={}):

    cocoeval = COCOeval(cocoGt=cocoGt,cocoDt=cocoDt, iouType="bbox")

    # Set custom parameters to cocoeval
    eval_params = cocoeval.params
    for attr in params.keys():
      setattr(eval_params, attr, params[attr])

    cocoeval.evaluate()
    cocoeval.accumulate()

    return cocoeval.eval



def get_mAP(eval, steps=21):
  T = eval['counts'][0] # Number of iou thresholds (n in formula)

  t0 = 0
  t50 = int((T - 1) / 2)
  t95 = int((T - 1) / 20 * 19)
  t100 = T - 1

  # Reshape recall array to [T, K, 1, A, M]
  # by adding a 3rd dimension to r_arr
  r_arr = np.expand_dims(eval['recall'], 2)

  # Reshape precision array to [T, K, R, A, M]
  # by switching 2nd and 3rd dimensions
  p_arr = np.transpose(eval['precision'], (0, 2, 1, 3, 4))

  # calculate mAP
  def _calc_mAP(t_start, t_end):
    # t_start, t_end = start and end IoU thresholds
    t_end += 1
    r_curve = r_arr[t_start:t_end, :, :, :, :]
    p_curve = p_arr[t_start:t_end, :, :, :, :]

    r_curve_diff = -np.diff(r_curve, axis=0, append=0)
    AP = np.sum(r_curve_diff * p_curve, axis=0)

    return np.average(AP, axis=0)

  # mAP_scores = [RxAxM] sized matrices containing mAP values
  mAP_scores = {
    'mAP': _calc_mAP(t0, t100), # mean Average Precision
    'mAP50': _calc_mAP(t50, t100),
    'mAP50-95': _calc_mAP(t50, t95)
  }

  return mAP_scores



def get_mR(eval):

  r_arr = eval['recall']

  r_arr = np.where(r_arr < 0, np.nan, r_arr)

  # mR_arr = [1xAxM] sized matrix containing mR values
  mR_arr = np.nanmean(r_arr, axis=(0, 1))

  mR_arr = np.expand_dims(mR_arr, 0)

  return mR_arr

import numpy as np



def get_mP(eval):

  p_arr = eval['precision']

  p_arr = np.where(p_arr < 0, np.nan, p_arr)

  # mP_arr = [RxAxM] sized matrices containing mP values
  mP_arr = np.nanmean(p_arr, axis=(0, 2))

  return mP_arr



def get_metrics(cocoGt, cocoDt, steps=21):

  if (steps - 1) % 20 != 0:
    raise Exception('(steps - 1) must be divisible by 20')

  params = {
    'areaRng': [[0, 1e5 ** 2]],
    'areaRngLbl': ['lahat na lods'],
    'recThrs': [1],
    'iouThrs': np.linspace(0, 1, steps),
    'maxDets': [100]
  }

  metrics = {}

  # Perform cocoeval
  eval = do_cocoeval(cocoGt, cocoDt, params)

  # Get recall, precision and F1
  R = get_mR(eval)[0, 0, 0].item()
  P = get_mP(eval)[0, 0, 0].item()
  metrics['R'] = R
  metrics['P'] = P
  metrics['F1'] = 2 * (P*R) / (P + R)

  # Get mAP scores
  mAP_scores = get_mAP(eval)
  metrics['mAP'] = mAP_scores['mAP'][0, 0, 0].item()
  metrics['mAP50'] = mAP_scores['mAP50'][0, 0, 0].item()
  metrics['mAP50-95'] = mAP_scores['mAP50-95'][0, 0, 0].item()

  return metrics



def get_ocr_metrics(coco_gt, coco_dt):
  images = coco_gt['images']
  gt_annots = coco_gt['annotations']
  dt_annots = coco_dt['annotations']

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  gt_img = torch.tensor([
    annot['image_id'] for annot in gt_annots
  ], dtype=torch.int, device=device)

  dt_img = torch.tensor([
    annot['image_id'] for annot in dt_annots
  ], dtype=torch.int, device=device)

  gt_ctg = torch.tensor([
    annot['category_id'] for annot in gt_annots
  ], dtype=torch.int, device=device)

  dt_ctg = torch.tensor([
    annot['category_id'] for annot in dt_annots
  ], dtype=torch.int, device=device)

  gt_img = gt_img.unsqueeze(1)
  dt_img = dt_img.unsqueeze(0)

  gt_ctg = gt_ctg.unsqueeze(1)
  dt_ctg = dt_ctg.unsqueeze(0)

  # Find the gt and dt annotations with common image_id and category_id
  same_img_and_ctg = (gt_img == dt_img) & (gt_ctg == dt_ctg)
  gt_dt_pairs = torch.nonzero(same_img_and_ctg, as_tuple=False)

  # Get similarity scores of texts between gt and dt
  similarities = [
    ratio(
      gt_annots[gt_dt_pair[0].item()]['text'],
      dt_annots[gt_dt_pair[1].item()]['text'],
    ) for gt_dt_pair in gt_dt_pairs
  ]
  similarities = torch.tensor(similarities, dtype=torch.float32, device=device)

  # aveSim = average levenshtein similarity
  aveSim = torch.mean(similarities).item()

  # For each annotation that exist in either gt and dt annotations but not both, add zero to similarities
  gt_zeros = torch.zeros(torch.sum(torch.all(same_img_and_ctg == False, dim=0)), dtype=torch.float32, device=device)
  dt_zeros = torch.zeros(torch.sum(torch.all(same_img_and_ctg == False, dim=1)), dtype=torch.float32, device=device)
  similarities = torch.cat((similarities, gt_zeros, dt_zeros), dim=0)

  # aveSimMiss = average levenshtein similarity including missing fields
  aveSimMiss = torch.mean(similarities).item()

  return {
    "Average Levenshtein Similarity": aveSim,
    "Average Levenshtein Similarity (Including missing fields)": aveSimMiss
  }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate detect evaluation metrics for given ground truth and detected truth COCO datasets"
    )

    parser.add_argument(
        "-g", "--coco-gt",
        required=True,
        help="Path to the ground truth COCO .json file"
    )

    parser.add_argument(
        "-d", "--coco-dt",
        required=True,
        help="Path to the detections COCO .json file"
    )

    args = parser.parse_args()

    with open(args.coco_gt, 'r') as f:
      coco_gt = json.load(f)

    with open(args.coco_dt, 'r') as f:
      coco_dt = json.load(f)

    cocoGt = COCO_from_dict(coco_gt)
    cocoDt = COCO_from_dict(coco_dt)

    metrics = get_metrics(cocoGt, cocoDt)

    print('\nMetrics for field appropriateness:')
    print(metrics)

    ocr_metrics = get_ocr_metrics(coco_gt, coco_dt)
    print('\nMetrics for ocr:')
    print(ocr_metrics)


if __name__ == "__main__":
    main()