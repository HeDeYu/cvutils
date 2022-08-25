# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/20 10:37

from typing import Tuple, Union

import cv2
import numpy as np

__all__ = [
    "pad_img",
    "resize_img",
    "put_fg_img_on_bg_img",
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


def put_fg_img_on_bg_img(fg_img, bg_img, top_left_xy=(0, 0), mask=None):
    assert len(fg_img.shape) == len(bg_img.shape)
    if len(fg_img.shape) == 3:
        assert fg_img.shape[2] == 3
        assert bg_img.shape[2] == 3

    h, w = fg_img.shape[:2]
    bottom_right_xy = top_left_xy[0] + w, top_left_xy[1] + h
    H, W = bg_img.shape[:2]
    assert bottom_right_xy[0] <= W
    assert bottom_right_xy[1] <= H
    if mask is None:
        if len(fg_img.shape) == 3:
            bg_img[
                top_left_xy[1] : bottom_right_xy[1],
                top_left_xy[0] : bottom_right_xy[0],
                :,
            ] = fg_img.copy()
            return bg_img
