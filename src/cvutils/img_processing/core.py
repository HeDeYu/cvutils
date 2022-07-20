# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/20 10:37

import numpy as np

from ..core import check_array

__all__ = [
    "get_mask_with_designated_color",
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
