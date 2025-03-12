import argparse

import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

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



def get_metrics(coco_gt, coco_dt, steps=21):

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
  eval = do_cocoeval(coco_gt, coco_dt, params)

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

    # cocoGt = COCO_from_dict(coco_gt)
    # cocoDt = COCO_from_dict(coco_dt)

    cocoGt = COCO(args.coco_gt)
    cocoDt = COCO(args.coco_dt)

    metrics = get_metrics(cocoGt, cocoDt)

    print('\n\nMetrics for adapted templates:')
    print(metrics)


if __name__ == "__main__":
    main()