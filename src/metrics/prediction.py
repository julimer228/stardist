import math
import numpy as np
from matplotlib import pyplot as plt


class Instance(object):
    def __init__(self, img_labeled, region):
        self.region = region
        self.img = (img_labeled == region.label).astype(int)
        self.area = region.area
        self.seen = False

    def get_aspect_ratio(self):
        min_row, min_col, max_row, max_col = self.region.bbox
        width = max_col - min_col
        height = max_row - min_row
        aspect_ratio = width / height
        return aspect_ratio

    def get_circularity(self):
        return (4 * math.pi * self.area) / (self.get_perimeter() ** 2)

    def get_perimeter(self):
        return self.region.perimeter


def get_iou(inst_1, inst_2):
    intersection = get_intersection(inst_1.img, inst_2.img)
    union = np.sum(np.logical_or(inst_1.img, inst_2.img))
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def get_intersection(mask_1, mask_2):
    return np.sum(np.logical_and(mask_1, mask_2))


def show_mask(mask_1, mask_2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(mask_1, cmap='gray')
    axes[0].set_title('mask_1')
    axes[0].axis('off')
    axes[1].imshow(mask_2, cmap='gray')
    axes[1].set_title('mask_2')
    axes[1].axis('off')
    plt.show()


def calculate_stats_per_img(gts, predictions, iou_thresh=0.75):
    """
    Calculate stats
    :param: gts: ground truth annotations
    :param: predictions: predicted annotations
    :param: iou_thresh: intersection over union threshold value
    :return: TP, FP and all positive values
    """

    TP = np.zeros(len(predictions))
    FP = np.zeros(len(predictions))
    IOUs = np.zeros(len(gts))
    for i in range(len(predictions)):
        iou_max = -1
        jMax = -1

        for j in range(len(gts)):
            iou = get_iou(predictions[i], gts[j])
            if iou < 0:
                raise ValueError("error")
            if iou > iou_max:
                iou_max = iou
                jMax = j
        if iou_max >= iou_thresh:
            if not gts[jMax].seen:
                # prediction with the highest IoU for given GT
                IOUs[jMax] = iou_max
                TP[i] = 1
                gts[jMax].seen = True
            else:
                # redundant predictions for given GT
                gts[jMax].seen = True
                FP[i] = 1
        else:
            # object detected in a place where it does not exist or detected inaccurately IoU<T
            FP[i] = 1

    all_positives = len(gts)
    return np.sum(TP), np.sum(FP), all_positives, np.sum(IOUs)


