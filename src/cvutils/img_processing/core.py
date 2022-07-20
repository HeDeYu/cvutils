# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/20 10:37

import cv2
import numpy as np

from ..core import check_array
from .polygon_roi import PolygonROI

__all__ = [
    "get_mask_with_designated_color",
    "get_cnt_items_by_color",
]


def get_mask_with_designated_color(mask, color):
    """

    Args:
        mask: provided mask image with dtype=np.uint8, ndim=3
        color: pixel with which color needed to assign 255, others 0

    Returns: ret with dtype=np.uint8, ndim=2, HW=mask.HW each pixel 255 or 0

    """
    check_array(dtype=np.uint8, ndim=3, mask=mask)
    ret = (mask == color).astype(np.uint8)
    temp = np.bitwise_and(ret[:, :, 0], ret[:, :, 1])
    ret = np.bitwise_and(temp, ret[:, :, 2])
    ret = ret * 255
    # check_array(dtype=np.uint8, ndim=2, ret=ret)
    return ret


def get_cnt_items_by_color(mask, color):
    """

    Args:
        mask: provided mask image with dtype=np.uint8, ndim=3
        color: region with which color needed to be get contour

    Returns: a list of PolygonROI_ instances

    """
    assert len(mask.shape) == 2 or len(mask.shape) == 3
    if color is not None:
        check_array(dtype=np.uint8, ndim=3, mask=mask)
        mask = get_mask_with_designated_color(mask=mask, color=color)
    else:
        # todo
        raise ValueError("color is not provided")

    # get contours，生成PolygonROI对象(扩展的PolygonROI对象，暂时实现在rraitools中，考虑更新到rrcvutils中
    cnts, _ = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    return PolygonROI.create_from_cv2_cnts(cnts)
