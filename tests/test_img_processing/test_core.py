# -*- coding:utf-8 -*-
# @FileName :test_core.py
# @Author   :Deyu He
# @Time     :2022/8/29 13:34

from unittest import TestCase

import numpy as np

from cvutils.img_processing import flip_img


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
        pass

    def tearDown(self) -> None:
        pass

    def test_flip_img(self):
        src = np.zeros((self.block_wh * 2, self.block_wh * 3, 3), dtype=np.uint8)
        src[: self.block_wh, self.block_wh : self.block_wh * 2, :] = (
            self.block[:, :, :] * 255
        )
        src[self.block_wh :, : self.block_wh, :] = self.block[:, :, :] * 255
        src[self.block_wh :, self.block_wh * 2 :, :] = self.block[:, :, :] * 128
        dst_v = flip_img(src, "vertical")  # noqa: F841
        dst_h = flip_img(src, "horizontal")  # noqa: F841
        dst_d = flip_img(src, "diagonal")  # noqa: F841
        # from cvutils import imshow
        # imshow(src, "ori", 0, 1)
        # imshow(dst_v, "vertical", 0, 1)
        # imshow(dst_h, "horizontal", 0, 1)
        # imshow(dst_d, "diagonal", 0, 1)
