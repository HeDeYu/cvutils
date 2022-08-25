# -*- coding:utf-8 -*-
# @FileName :utils.py
# @Author   :Deyu He
# @Time     :2022/8/25 15:29

import cv2
import numpy as np

from .polygon_roi import PolygonROI

__all__ = [
    "transform_img",
]


def transform_img(img, rotation_deg, scale=1.0, return_mask=True):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center=(0.0, 0.0), angle=rotation_deg, scale=scale)
    v = np.array(
        [
            [0, 0, 1],
            [0, h, 1],
            [w, h, 1],
            [w, 0, 1],
        ]
    )
    v_new_t = np.matmul(M, v.transpose())

    x_min, y_min = np.min(v_new_t, axis=1)
    x_max, y_max = np.max(v_new_t, axis=1)
    w_new = round(x_max - x_min) + 1
    h_new = round(y_max - y_min) + 1
    M[:, 2] += np.array([-x_min, -y_min])

    img_ret = cv2.warpAffine(img, M, (w_new, h_new))

    if return_mask:
        mask = np.zeros((h_new, w_new), dtype=np.uint8)
        PolygonROI(v_new_t.transpose() + np.array([-x_min, -y_min])).draw(
            mask, color=255, thickness=-1
        )
        return img_ret, M, mask
    else:
        return img_ret, M, None
