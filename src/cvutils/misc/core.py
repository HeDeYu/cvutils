# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/8/11 11:05

import cv2
import numpy as np

from ..core.core import check_array, check_array_dtype
from ..img_processing.polygon_roi import PolygonROI
from ..img_processing.rect_roi import RectROI

__all__ = [
    "draw_bboxes_on_img",
    "get_cnt_items_by_color",
    "get_mask_with_designated_color",
    "draw_cnts_on_img_by_colored_mask",
    "draw_cnts_on_img_by_natural_coding_mask",
    "convert_natural_coding_to_mask",
    "cal_pts_fn_fp",
    "cal_hit_num",
    "cal_components_hit_num",
    "cal_components_fn_fp",
    "filter_base_func",
    "remove_cnts_with_undersized_area",
    "remove_polygon_rois_with_undersized_area",
]


def convert_natural_coding_to_mask(natural_coding, color_list):
    mask = np.zeros(
        (natural_coding.shape[0], natural_coding.shape[1], 3), dtype=np.uint8
    )
    for idx, color in enumerate(color_list):
        mask[np.where(cv2.inRange(natural_coding, idx, idx))] = color
    return mask


def get_mask_with_designated_color(mask, color):
    """
    给定单通道或三通道的mask以及对应的color（单一int值或者color），返回由符合该color的区域组成的mask图
    :param mask: provided mask image with dtype=np.uint8, ndim=3 or 2
    :param color: pixel with which color needed to assign 255, others 0
    :return: ret with dtype=np.uint8, ndim=2, HW=mask.HW each pixel with value of 255 or 0
    """
    return cv2.inRange(mask, color, color)


def get_cnt_items_by_color(mask, color):
    """
    给定单通道或三通道的mask以及color，返回该mask中该color的轮廓列表，轮廓以本库的PolygonROI对象表达。
    :param mask: np.ndarray, provided mask image with dtype=np.uint8, ndim=3 or 2
    :param color: region with which color needed to be find contours
    :return: a list of PolygonROI instances
    """
    assert len(mask.shape) == 2 or len(mask.shape) == 3
    if isinstance(color, (list, tuple)):
        assert len(color) == 3
        check_array(dtype=np.uint8, ndim=3, mask=mask)
    elif isinstance(color, int):
        check_array(dtype=np.uint8, ndim=2, mask=mask)
    else:
        # todo
        raise ValueError("color is not provided")
    mask = get_mask_with_designated_color(mask=mask, color=color)
    cnts, _ = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    return PolygonROI.create_from_cv2_cnts(cnts)


def draw_cnts_on_img_by_colored_mask(img, mask, palette, thickness=-1):
    """

    :param img:
    :param mask:
    :param color_list:
    :param thickness:
    :return:
    """
    for color in palette:
        if color == [0, 0, 0]:
            # logger.debug("bg!")
            continue
        polygon_roi_list = get_cnt_items_by_color(mask, color)
        for polygon_roi in polygon_roi_list:
            polygon_roi.draw(img, color, thickness=thickness)
    return img


def draw_cnts_on_img_by_natural_coding_mask(img, mask, palette, thickness=-1):
    """

    :param img:
    :param mask:
    :param palette:
    :param thickness:
    :return:
    """
    colored_mask = np.zeros_like(img, dtype=np.uint8)
    for idx, color in enumerate(palette):
        colored_mask[np.where(cv2.inRange(mask, idx, idx))] = color
    img_ret = draw_cnts_on_img_by_colored_mask(
        img.copy(), colored_mask, palette, thickness=thickness
    )
    return img_ret


def draw_bboxes_on_img(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            RectROI.create_from_xyxy(*bbox).draw(img, color, thickness=1)
    return img


def cal_components_hit_num(mask_1, mask_2, overlap_ratio_th=0):
    """

    Args:
        mask_1: the 8-bit single-channel image
        mask_2: the 8-bit single-channel image
        overlap_ratio_th: hit_ratio(the intersect of component B in mask_2 component A in mask_1 divided by the area of component A) threshold

    Returns:
        1st: the number of the components in mask_1 are hit by mask_2
        2nd: the total number of components in mask_1
    """
    check_array_dtype(np.uint8, mask_1=mask_1, mask_2=mask_2)
    check_array(ndim=2, mask_1=mask_1, mask_2=mask_2)
    hit_component_num = 0
    num_components, xs_map, stats, _ = cv2.connectedComponentsWithStats(mask_1)
    hit_map = xs_map * (mask_2 > 0)
    for idx_component in range(1, num_components):
        hit_pixel_num = np.count_nonzero(hit_map == idx_component)
        components_area = stats[idx_component][-1]
        if hit_pixel_num > 0 and hit_pixel_num / components_area > overlap_ratio_th:
            hit_component_num += 1
    return hit_component_num, num_components - 1


def cal_components_fn_fp(
    pred_mask, gt_mask, fn_overlap_ratio_th=0.0, fp_overlap_ratio_th=0.0
):
    """

    Args:
        pred_mask: the 8-bit single-channel image of predicted mask
        gt_mask: the 8-bit single-channel image of ground truth mask
        overlap_ratio_th: see cal_components_hit_num

    Returns:
        #false negative, #false positive, #groud truth, #prediction
    """
    gt_hit_num, gt_num = cal_components_hit_num(
        mask_1=gt_mask, mask_2=pred_mask, overlap_ratio_th=fn_overlap_ratio_th
    )
    pred_hit_num, pred_num = cal_components_hit_num(
        mask_1=pred_mask, mask_2=gt_mask, overlap_ratio_th=fp_overlap_ratio_th
    )
    return gt_num - gt_hit_num, pred_num - pred_hit_num, gt_num, pred_num


def cal_hit_num(xs, ys, min_dist):
    hit_num = 0
    if len(ys) == 0:
        return hit_num
    for x in xs:
        dists = np.sqrt(np.sum((ys - x) ** 2, axis=1))
        _min_dist = np.min(dists)
        if _min_dist <= min_dist:
            hit_num += 1
    return hit_num


def cal_pts_fn_fp(pred_pts, gt_pts, fn_min_dist, fp_min_dist):
    assert (
        gt_pts.shape[1] == pred_pts.shape[1]
    ), "gt_pts should have same point shape as pred_pts"

    gt_hit_num = cal_hit_num(gt_pts, pred_pts, fn_min_dist)
    pred_hit_num = cal_hit_num(pred_pts, gt_pts, fp_min_dist)
    gt_num = len(gt_pts)
    pred_num = len(pred_pts)

    return gt_num, pred_num, gt_num - gt_hit_num, pred_num - pred_hit_num


def filter_base_func(
    mask, color, mask_ret=None, filter_func_on_polygon_rois=None, **kwargs
):
    """
    给定uint8单通道或三通道的mask，对应的color（int或list/tuple），返回目标图像（若不给定，则生成一张与mask dims一致的全黑图片作为返回目标图像），过滤方法以及过滤方法的参数，
    在mask中提取color区域的轮廓列表，轮廓以本库的PolygonROI对象表达，该轮廓列表经由过滤方法过滤后，保留的轮廓以给定的color绘制在返回目标图像上。
    :param mask: provided mask image with dtype=np.uint8, ndim=3 or 2, contours are detected on this image
    :param color: region with which color will be detected contours
    :param mask_ret: filtered result (contours) will draw on this image, if not provided, it will init as zeros with shape = mask.shape
    :param filter_func_on_polygon_rois: filter function
    :param kwargs: kwargs for above filter function
    :return: the image on which the filtered contours (region with given color in mask) drawn
    """
    if mask_ret is None:
        mask_ret = np.zeros_like(mask)
    cnt_items = get_cnt_items_by_color(mask, color)
    if filter_func_on_polygon_rois is not None:
        cnt_items_remained = filter_func_on_polygon_rois(cnt_items, **kwargs)
    else:
        cnt_items_remained = cnt_items
    if isinstance(color, int):
        color = 255
    for cnt_item in cnt_items_remained:
        cnt_item.draw(mask_ret, color=color, thickness=-1)
    return mask_ret


def remove_polygon_rois_with_undersized_area(cnt_items, area_th):
    """

    :param cnt_items:
    :param area_th:
    :return:
    """
    return list(
        filter(
            lambda cnt_item: cnt_item.area > area_th,
            cnt_items,
        )
    )


def remove_cnts_with_undersized_area(mask, color, mask_ret, area_th):
    """

    :param mask:
    :param color:
    :param mask_ret:
    :param area_th:
    :return:
    """
    return filter_base_func(
        mask, color, mask_ret, remove_polygon_rois_with_undersized_area, area_th=area_th
    )
