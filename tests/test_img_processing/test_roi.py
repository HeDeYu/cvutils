# -*- coding:utf-8 -*-
# @FileName :test_roi.py
# @Author   :Deyu He
# @Time     :2022/8/25 11:09

from unittest import TestCase

import numpy as np

from cvutils.img_processing import PolygonROI, RectROI


class TestPolygonROI(TestCase):
    def setUp(self) -> None:
        poly1 = np.array(
            [
                [0, 0],
                [100, 0],
                [100, 100],
                [0, 100],
            ]
        )
        self.poly1 = PolygonROI(poly1)

        poly2 = np.array(
            [
                [50, 20],
                [120, 20],
                [120, 120],
                [50, 120],
            ]
        )
        self.poly2 = PolygonROI(poly2)

        poly3 = np.array(
            [
                [100, 0],
                [120, 0],
                [120, 19],
                [100, 19],
            ]
        )
        self.poly3 = PolygonROI(poly3)

    def tearDown(self) -> None:
        pass

    def test_pixel_area(self):
        assert self.poly1.pixel_area == 10201
        assert self.poly2.pixel_area == 7171
        assert self.poly3.pixel_area == 420

    def test_iou_and_overlap(self):
        iou, i, u = self.poly1.iou(self.poly2)
        assert i == 51 * 81
        assert u == 10201 + 7171 - 51 * 81
        assert iou == float(i) / float(u)
        assert self.poly1.overlap(self.poly2)

        iou, i, u = self.poly2.iou(self.poly3)
        assert i == 0
        assert u == 7171 + 420
        assert iou == 0
        assert not self.poly2.overlap(self.poly3)

        iou, i, u = self.poly3.iou(self.poly1)
        assert i == 20
        assert u == 10201 + 420 - 20
        assert self.poly3.overlap(self.poly1)

    def test_create_from_rect_xyxy(self):
        pass


class TestRectROI(TestCase):
    def setUp(self) -> None:
        rect1 = [
            0,
            0,
            # [100, 0],
            100,
            100,
            # [0, 100],
        ]
        self.rect1 = RectROI.create_from_xyxy(*rect1)

        rect2 = [
            50,
            20,
            # [120, 20],
            120,
            120,
            # [50, 120],
        ]

        self.rect2 = RectROI.create_from_xyxy(*rect2)

        rect3 = [
            100,
            0,
            # [120, 0],
            120,
            19,
            # [100, 19],
        ]
        self.rect3 = RectROI.create_from_xyxy(*rect3)

    def tearDown(self) -> None:
        pass

    # def test_temp(self):
    #     img = np.zeros((200, 200))
    #     import cv2
    #     cv2.rectangle(img, (0, 0), (100, 100), color=1, thickness=-1)
    #     print(cv2.countNonZero(img))
