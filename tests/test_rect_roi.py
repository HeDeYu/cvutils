# -*- coding:utf-8 -*-
# @FileName :test_rect_roi.py
# @Author   :Deyu He
# @Time     :2022/11/29 16:48
from unittest import TestCase

import cv2
import numpy as np

import cvutils


class TestRectROI(TestCase):
    def test_transform(self):
        img = np.ones((256, 256, 3), dtype=np.uint8) * 128

        rect = cvutils.RectROI.create_from_xywh(0, 0, 100, 100)
        _ = rect.crop(img)
        # cvutils.imshow(rect_img, "roi", 0)

        rect.draw(img, (0, 0, 255), 1)

        rect.transform(translation=(100, 100))
        rect.draw(img, (255, 0, 0), 1)
        # cvutils.imshow(img, "ori", 0)

    def test_bounding_rect(self):
        pts = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
        print(cv2.boundingRect(pts))
