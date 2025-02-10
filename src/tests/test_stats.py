import unittest

import numpy as np
from skimage.measure import label, regionprops

from src.metrics.prediction import get_intersection, Instance, get_iou


class TestStats(unittest.TestCase):

    def test_get_intersection(self):
        inst_1 = np.full((10, 10), 0)
        inst_2 = np.full((10, 10), 0)
        inst_3 = np.full((10, 10), 0)

        inst_1[0:2, :] = 1
        inst_2[7:9, :] = 1
        inst_3[8:9, :] = 1

        self.assertEqual(0, get_intersection(inst_1, inst_2))
        self.assertEqual(10, get_intersection(inst_3, inst_2))

    def test_get_iou(self):
        img_1 = np.full((10, 10), 0)
        img_2 = np.full((10, 10), 0)
        img_3 = np.full((10, 10), 0)

        img_1[0:2, :] = 1
        img_2[7:9, :] = 1
        img_3[8:9, 1:9] = 1
        print(img_3)

        img_1_labelled = label(img_1)
        img_2_labelled = label(img_2)
        img_3_labelled = label(img_3)

        inst_1 = Instance(img_1_labelled, regionprops(img_1_labelled)[0])
        inst_2 = Instance(img_2_labelled, regionprops(img_2_labelled)[0])
        inst_3 = Instance(img_3_labelled, regionprops(img_3_labelled)[0])

        self.assertEqual(0/40, get_iou(inst_1, inst_2))
        self.assertEqual(8/20, get_iou(inst_3, inst_2))



if __name__ == '__main__':
    unittest.main()