# -*- coding:utf-8 -*-
# @FileName :test_utils.py
# @Author   :Deyu He
# @Time     :2022/8/25 17:43

# import cv2
import numpy as np

# import cvutils
from cvutils.img_processing import transform_img

img = np.ones((256, 512, 3), dtype=np.uint8) * 128

img_transformed, M, mask = transform_img(img, rotation_deg=30, scale=1.5)
# cvutils.imshow(img_transformed, "ret", 0, flags=cv2.WINDOW_AUTOSIZE)
# cvutils.imshow(mask, "mask", 0, flags=cv2.WINDOW_AUTOSIZE)
