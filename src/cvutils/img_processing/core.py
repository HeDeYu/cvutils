# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/20 10:37

from typing import Tuple, Union

import cv2
import numpy as np

from ..core import check_array
from .polygon_roi import PolygonROI

__all__ = [
    "pad_img",
    "resize_img",
    "get_mask_with_designated_color",
    "get_cnt_items_by_color",
]


def resize_img(
    img, dsize_wh, keep_aspect_ratio=False, interpolation=cv2.INTER_NEAREST
) -> np.ndarray:
    """Resize an image into `dsize`.

    Args:
        img(array_like): A ndarray.
        dsize_wh(tuple): Image size after resize.
        keep_aspect_ratio(bool): Whether keep the ratio of width and height.
        interpolation(int): Interpolation method of resize.

    Returns:
        np.ndarray: The resized image.

    """

    if not keep_aspect_ratio:
        resized_img = cv2.resize(img, dsize_wh, interpolation=interpolation)
    else:
        img_h, img_w = img.shape[:2]
        scale = min(dsize_wh[1] / img_h, dsize_wh[0] / img_w)
        resized_img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=interpolation
        )
    return resized_img


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

    cnts, _ = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    return PolygonROI.create_from_cv2_cnts(cnts)


def pad_img(
    img: np.ndarray,
    dsize_wh: Union[Tuple[int, int], None] = None,
    pad_tblr=(0, 0, 0, 0),
    border_type=cv2.BORDER_CONSTANT,
    fill_value: Union[Tuple[int, int, int], int] = (255, 255, 255),
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad an image with `border_type`. If `border_type` is cv2.BORDER_CONSTANT, padded area will
    be filled by `fill_value`.
        If `dsize_wh` is not None, img will be centered on padded_img, otherwise img will be padded
    by `pad_tblr`.

    Args:
        img(array_like): A ndarray.
        dsize_wh(tuple): Image size after padded.
        pad_tblr(tuple): The area of the top, bottom, left, right to the source image.
        border_type(int): Interpolation method, Refers to doc of opencv, `BorderTypes` for more details.
        fill_value(tuple): Fill the value when border_type is cv2.BORDER_CONSTANT

    Returns:
        np.ndarray: The padded image.
        tuple: The area of the top, bottom, left, right to the source image.
    """

    if dsize_wh is not None:
        src_h, src_w = img.shape[:2]
        dst_w, dst_h = dsize_wh

        pad_l = int((dst_w - src_w) / 2)
        pad_r = dst_w - src_w - pad_l
        pad_t = int((dst_h - src_h) / 2)
        pad_b = dst_h - src_h - pad_t
        pad_tblr = (pad_t, pad_b, pad_l, pad_r)
    else:
        pad_t, pad_b, pad_l, pad_r = pad_tblr

    padded_img = cv2.copyMakeBorder(
        img,
        top=pad_t,
        bottom=pad_b,
        left=pad_l,
        right=pad_r,
        borderType=border_type,
        value=fill_value,
    )
    return padded_img, pad_tblr
