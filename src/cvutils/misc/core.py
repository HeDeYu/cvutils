# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/8/11 11:05

import cv2
import numpy as np

from ..core.core import check_array
from ..img_processing.polygon_roi import PolygonROI
from ..img_processing.rect_roi import RectROI

__all__ = [
    "draw_bboxes_on_img",
    "get_cnt_items_by_color",
    "get_mask_with_designated_color",
    "draw_cnts_on_img_by_colored_mask",
    "draw_cnts_on_img_by_natural_coding_mask",
    "convert_natural_coding_to_mask",
]


def draw_bboxes_on_img(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            RectROI.create_from_xyxy(*bbox).draw(img, color, thickness=1)
    return img


def get_mask_with_designated_color(mask, color, max_val=255):
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
    ret = ret * max_val
    # check_array(dtype=np.uint8, ndim=2, ret=ret)
    return ret


def get_cnt_items_by_color(mask, color):
    """

    Args:
        mask: provided mask image with dtype=np.uint8, ndim=3
        color: region with which color needed to be get contour

    Returns: a list of PolygonROI instances

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


def draw_cnts_on_img_by_colored_mask(img, mask, color_list, thickness=-1):
    for color in color_list:
        if color == [0, 0, 0]:
            # logger.debug("bg!")
            continue
        polygon_roi_list = get_cnt_items_by_color(mask, color)
        for polygon_roi in polygon_roi_list:
            polygon_roi.draw(img, color, thickness=thickness)
    return img


def draw_cnts_on_img_by_natural_coding_mask(img, mask, palette, thickness=-1):
    colored_mask = np.zeros_like(img, dtype=np.uint8)
    for idx, color in enumerate(palette):
        colored_mask[np.where(cv2.inRange(mask, idx, idx))] = color
    img_ret = draw_cnts_on_img_by_colored_mask(
        img.copy(), colored_mask, palette, thickness=thickness
    )
    return img_ret


def convert_natural_coding_to_mask(natural_coding, color_list):
    mask = np.zeros(
        (natural_coding.shape[0], natural_coding.shape[1], 3), dtype=np.uint8
    )
    for idx, color in enumerate(color_list):
        mask[np.where(cv2.inRange(natural_coding, idx, idx))] = color
    return mask
