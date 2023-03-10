# -*- coding:utf-8 -*-
# @FileName :test_core.py
# @Author   :Deyu He
# @Time     :2022/8/29 13:34

import copy
from unittest import TestCase

import numpy as np

from cvutils.img_processing import (
    flip_img,
    flip_img_with_rois,
    flip_pts,
    rotate_and_scale_img,
)


class TestCore(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        setattr(TestCore, "block_wh", 256)
        setattr(
            TestCore,
            "block",
            np.ones(
                (getattr(TestCore, "block_wh"), getattr(TestCore, "block_wh"), 3),
                dtype=np.uint8,
            ),
        )

    def setUp(self) -> None:
        src = np.zeros(
            (getattr(self, "block_wh") * 2, getattr(self, "block_wh") * 3, 3),
            dtype=np.uint8,
        )
        src[
            : getattr(self, "block_wh"),
            getattr(self, "block_wh") : getattr(self, "block_wh") * 2,
            :,
        ] = (
            getattr(self, "block")[:, :, :] * 255
        )
        src[getattr(self, "block_wh") :, : getattr(self, "block_wh"), :] = (
            getattr(self, "block") * 255
        )
        src[getattr(self, "block_wh") :, getattr(self, "block_wh") * 2 :, :] = (
            getattr(self, "block")[:, :, :] * 128
        )
        self.src = src
        self.flip_flags = ["vertical", "horizontal", "diagonal"]
        self.pts = np.array(
            [
                [getattr(self, "block_wh") * 0.5, getattr(self, "block_wh") * 1.5],
                [getattr(self, "block_wh") * 2.0, 0.0],
                [getattr(self, "block_wh") * 2.5, getattr(self, "block_wh") * 0.5],
                [getattr(self, "block_wh") * 1.5, getattr(self, "block_wh") * 1.8],
            ]
        )

    def tearDown(self) -> None:
        pass

    def test_flip_img(self):
        src = copy.deepcopy(self.src)
        for flip_flag in self.flip_flags:
            dst = flip_img(src, flip_flag)  # noqa: F841

            # from cvutils import imshow
            #
            # imshow(src, "ori", 0, 1)
            # imshow(dst, flip_flag, 0, 1)

    def test_flip_pts(self):
        src = copy.deepcopy(self.src)
        src_pts = self.pts
        for flip_flag in self.flip_flags:
            dst_pts = flip_pts(src_pts, src.shape[:2], flip_flag)  # noqa: F841

            # from cvutils import PolygonROI, imshow
            # src_pts_polygon = PolygonROI(src_pts)
            # dst_pts_polygon = PolygonROI(dst_pts)
            #
            # src_with_pts = src.copy()
            # src_pts_polygon.draw(src_with_pts, color=(0, 0, 255), thickness=1)
            # dst_pts_polygon.draw(src_with_pts, color=(255, 0, 0), thickness=1)
            # imshow(src_with_pts, flip_flag, 0, 1)

    def test_filp_img_with_rois(self):
        src = copy.deepcopy(self.src)
        src_pts = self.pts
        for flip_flag in self.flip_flags:
            dst, dst_pts = flip_img_with_rois(src, [src_pts], flip_flag)  # noqa: F841
            # from cvutils import PolygonROI, imshow
            # src_pts_polygon = PolygonROI(src_pts)
            # dst_pts_polygon = PolygonROI(dst_pts)
            # src_with_pts = src.copy()
            # src_pts_polygon.draw(src_with_pts, color=(0, 0, 255), thickness=1)
            # dst_pts_polygon.draw(dst, color=(255, 0, 0), thickness=1)
            # imshow(src_with_pts, "src", 0, 1)
            # imshow(dst, flip_flag, 0, 1)

    def test_rotate_and_scale_img(self):
        src = copy.deepcopy(self.src)
        dst_p45s1, _, mask_p45s1 = rotate_and_scale_img(
            src, rotation_angle_degree=45.0, return_mask=True
        )
        dst_n45s1, _, mask_n45s1 = rotate_and_scale_img(
            src, rotation_angle_degree=-45.0, return_mask=True
        )
        dst_p30s1, _, mask_p30s1 = rotate_and_scale_img(
            src, rotation_angle_degree=30.0, return_mask=True
        )
        dst_p30s1f5, _, mask_p30s1f5 = rotate_and_scale_img(
            src, rotation_angle_degree=30.0, scale=1.5, return_mask=True
        )
        dst_p30s0f5, _, mask_p30s0f5 = rotate_and_scale_img(
            src, rotation_angle_degree=30.0, scale=0.5, return_mask=True
        )
        # from cvutils import imshow
        # imshow(src, "ori", 0, 1)
        # imshow(dst_p45s1, "dst", 0, 1)
        # imshow(mask_p45s1, "mask", 0, 1)
        # imshow(dst_n45s1, "dst", 0, 1)
        # imshow(mask_n45s1, "mask", 0, 1)
        # imshow(dst_p30s1, "dst", 0, 1)
        # imshow(mask_p30s1, "mask", 0, 1)
        # imshow(dst_p30s1f5, "dst", 0, 1)
        # imshow(mask_p30s1f5, "mask", 0, 1)
        # imshow(dst_p30s0f5, "dst", 0, 1)
        # imshow(mask_p30s0f5, "mask", 0, 1)
