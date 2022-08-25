# -*- coding:utf-8 -*-
# @FileName :test_polygon_roi.py
# @Author   :Deyu He
# @Time     :2022/8/25 11:09

import numpy as np

from cvutils.img_processing import PolygonROI

poly1 = np.array(
    [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
    ]
)

poly2 = np.array(
    [
        [50, 20],
        [100, 20],
        [100, 100],
        [50, 100],
    ]
)
poly1 = PolygonROI(poly1)
poly2 = PolygonROI(poly2)
print(poly1.overlap(poly2))

poly1 = np.array(
    [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
    ]
)

poly2 = np.array(
    [
        [101, 20],
        [105, 20],
        [105, 100],
        [101, 100],
    ]
)
poly1 = PolygonROI(poly1)
poly2 = PolygonROI(poly2)
print(poly1.overlap(poly2))
