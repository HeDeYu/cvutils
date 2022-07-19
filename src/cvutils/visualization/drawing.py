# -*- coding:utf-8 -*-
# @FileName :drawing.py
# @Author   :Deyu He
# @Time     :2022/7/19 20:14

import cv2
import numpy as np

__all__ = [
    "draw_contours",
]


def draw_contours(
    img: np.ndarray,
    contours,
    contour_idx,
    color,
    thickness=1,
    lineType=8,
    hierarchy=None,
    maxLevel=1 << 31 - 1,
    offset=(0, 0),
):
    if hierarchy is not None:
        cv2.drawContours(
            image=img,
            contours=contours,
            contourIdx=contour_idx,
            color=color,
            thickness=thickness,
            lineType=lineType,
            hierarchy=hierarchy,
            maxLevel=maxLevel,
            offset=offset,
        )
    else:
        cv2.drawContours(
            image=img,
            contours=contours,
            contourIdx=contour_idx,
            color=color,
            thickness=thickness,
            lineType=lineType,
            offset=offset,
        )
