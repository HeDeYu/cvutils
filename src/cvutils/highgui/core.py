# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/11 20:18

import cv2

__all__ = [
    "waitkey",
    "imshow",
]


def waitkey(waittime_ms: int) -> int:
    """Waits for a pressed key

    This func is an overwrite of cv2.waitKey.
    In cv2.waitKey, when `waittime_ms` < 0, waitKey will blocked until a key pressed,
    while in this func, it will return immediate when `waittime_ms` < 0.

    Args:
        waittime_ms: waiting time in milliseconds.
            0 is the special value that means "forever",
            and negative value(<0) means "no wait"

    Returns:
        int: the code of the pressed key or -1 if no key was pressed
            before the specified time had elapsed.
    """
    return -1 if waittime_ms < 0 else cv2.waitKey(waittime_ms)


def imshow(
    img, win_name="image", wait_key_ms: int = -1, flags=cv2.WINDOW_NORMAL
) -> int:
    """Show image in an opencv window

    Create a window with specified `win_name`, show `img` by this window
    and waiting a specified time `wait_key_ms` for key press, return the
    ASCII code of pressed key.

    Args:
        img(np.ndarray): Shown image
        win_name(str): Title of shown window, default is "image"
        wait_key_ms(float): Time of waiting for key press in milliseconds.
            Positive value( > 0 ) means time of waiting,
            0 is the special value that means "forever", and
            negative ( < 0, default) means no waiting
        flags(int): Flags of the window
            Refer to opencv.namedWindow docs for more information

    Returns:
        int: ASCII code of pressed key, -1 if `wait_key_ms` is negative, which
            means no waiting for key press
    """
    cv2.namedWindow(win_name, flags)
    cv2.imshow(win_name, img)

    return waitkey(wait_key_ms)
