# -*- coding:utf-8 -*-
# @FileName :test_core.py
# @Author   :Deyu He
# @Time     :2022/8/29 13:34

from unittest import TestCase

import numpy as np

# from cvutils.img_processing import flip_img, rotate_and_scale_img


class TestCore(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        setattr(TestCore, "block_wh", 256)
        setattr(
            TestCore,
            "block_255",
            np.ones(
                (getattr(TestCore, "block_wh"), getattr(TestCore, "block_wh"), 3),
                dtype=np.uint8,
            )
            * 255,
        )

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_flip_img(self):
        src = np.zeros((self.block_wh * 2, self.block_wh * 3, 3), dtype=np.uint8)
        src[: self.block_wh, self.block_wh : self.block_wh * 2, :] = self.block_255[
            :, :, :
        ]
        src[self.block_wh :, : self.block_wh, :] = self.block_255[:, :, :]
        src[self.block_wh :, self.block_wh * 2 :, :] = self.block_255[:, :, :]
        from cvutils import imshow

        imshow(src, "src", 0, 1)
