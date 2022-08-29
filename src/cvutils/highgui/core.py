# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/11 20:18

import math
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pyutils

from ..img_processing.core import pad_img, resize_img

__all__ = [
    "waitkey",
    "imshow",
    "add_title_to_img",
    "grid_images",
    "imread",
    "imwrite",
]


def imread(filename, flags=cv2.IMREAD_UNCHANGED, raise_if_failed=False) -> np.ndarray:
    """Read image from file

    Args:
        filename(str, Path): filename of reading image
        flags(int): Flag that can take values of cv2::ImreadModes
            Refers to opencv.imread for more details
        raise_if_failed(bool): Raise exception if failed or not
            For cv2.imread, a None object will be returned if read failed.
            If set, an exception will be raised
            instead of return None.

    Returns:
        np.ndarray: reading image

    Raises:
        RRCVException: If `raise_if_failed` is set, and reading process failed

    """
    filename = Path(filename)

    img = cv2.imread(filename.as_posix(), flags)
    read_success = isinstance(img, np.ndarray)

    if not read_success and raise_if_failed:
        raise Exception(f"read image {filename} failed!")

    return img


def imwrite(filename, img, params=None) -> bool:
    """Write image to file

    Args:
        filename(str, Path): filename of writing image
        img(np.ndarray): writing image
        params: Format-specific parameters
            Refers to opencv.imwrite for more details

    Returns:
        bool: Write image success or failed

    """
    filename = Path(filename)

    write_success = cv2.imwrite(filename.as_posix(), img, params)

    return write_success


def add_title_to_img(
    img,
    title,
    horizontal_alignment="mid",
    vertical_alignment="top",
    avoid_decorate_axes=True,
    color: Union[int, Tuple] = (0, 0, 0),
    thickness=1,
    font_scale=1.0,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    face_color: Union[Tuple[int, int, int], int, None] = None,
    fill_color: Union[Tuple[int, int, int], int] = (255, 255, 255),
    line_type=cv2.LINE_8,
):
    """Add title to image at the top of center.

    Args:
        img(np.ndarray): Image.
        title(str): Title string to be drawn.
        horizontal_alignment(str): Horizontal alignment of the title, {'left', 'mid', 'right'}.
        vertical_alignment(str): Vertical alignment of the title, {'top', 'center', 'bottom'}.
        avoid_decorate_axes(bool): If is True,
             horizon of the title is determined automatically to avoid decorators on the Axes.
        color(tuple): Text color. Refer to opencv.
        thickness(int): Thickness of the lines used to draw a text. Refer to opencv.
        font_scale(float): Font scale. Refer to opencv.
        font_face(int): Font type. Refer to opencv.
        face_color(tuple): Face color of the title.
        fill_color(tuple): Filled color when avoid_decorate_axes is True.
        line_type(int): Line type. Refer to opencv.

    Returns:
        np.ndarray

    """
    # check location key
    pyutils.check_in_list(["left", "mid", "right"], loc=horizontal_alignment)
    pyutils.check_in_list(["top", "center", "bottom"], loc=vertical_alignment)

    # height of the text = title_h + baseline
    (title_w, title_h), baseline = cv2.getTextSize(
        title,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness,
    )

    if avoid_decorate_axes:

        if "top" == vertical_alignment:
            img, _ = pad_img(
                img, pad_tblr=(title_h + baseline, 0, 0, 0), fill_value=fill_color
            )

        elif "bottom" == vertical_alignment:
            img, _ = pad_img(
                img, pad_tblr=(0, title_h + baseline, 0, 0), fill_value=fill_color
            )

    img_h, img_w = img.shape[:2]

    # x_loc is the left of the text.
    x_loc_dict = {
        "left": 0,
        "mid": int((img_w - title_w) / 2 + 0.5),
        "right": img_w - title_w,
    }

    # y_loc if the bottom of the text.
    # 'Bottom' is the bottom of 'a', not the bottom of 'g'
    y_loc_dict = {
        "top": 0 + title_h,
        "center": int((img_h - title_h) / 2 + 0.5 + title_h),
        "bottom": img_h - baseline,
    }

    x_loc = x_loc_dict[horizontal_alignment]
    y_loc = y_loc_dict[vertical_alignment]

    if face_color is not None:
        if (img.ndim == 2) and isinstance(face_color, tuple):
            face_color = face_color[0]
        img[
            max(0, y_loc - title_h) : y_loc + baseline, max(0, x_loc) : x_loc + title_w
        ] = face_color

    cv2.putText(
        img,
        title,
        (x_loc, y_loc),
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness,
        color=color,
        lineType=line_type,
    )

    return img


def grid_images(
    imgs: List[np.ndarray],
    sub_plot_nrow_ncol=(1, -1),
    sub_plot_dict=None,
    titles: Union[List[str], None] = None,
    title_scale=0.6,
    title_thickness=1,
    title_font_dict=None,
) -> np.ndarray:
    """Show a batch of images in an opencv window

        Grid image with 'imgs'.

    Args:
        imgs(list of np.ndarray): Shown images.
        sub_plot_nrow_ncol(tuple): Merge format.
            value = (1, -1), image will be arranged horizontally.
            value = (-1, 1), image will be arranged vertically.
            value = (n, m), n!=-1 & m!=-1, n*m >= len(imgs), images will be arranged in n rows and m columns.
        sub_plot_dict(dict): A dictionary controlling the appearance of each subplot:
            size_wh(tuple): default=(128, 128), size of subplot.
            space_tblr(tuple): default=(0, 0, 0, 0), the amount of width/height reserved for space between subfigures.
            fill_color(tuple of int): default=(255, 255, 255), set color of filled space.
            spine_thickness(int): default=0, thickness of axis spine.
            spine_color(tuple of int): default=(0, 0, 0), set color of spine.
        titles(list): Titles of each image.
        title_scale(float): Set title size.
        title_thickness(int): Set title thickness.
        title_font_dict(dict): A dictionary controlling the appearance of the title text:
            font_face(int): default=cv2.FONT_HERSHEY_SIMPLEX, refer to add_title_to_img.
            horizontal_alignment(str): default='mid', set horizontal alignment of the title. Refer to add_title_to_img.
            vertical_alignment(str): default='top', set vertical alignment of the title. Refer to add_title_to_img.
            color(int or tuple of int): default=(0, 0, 0), set color of title. Refer to add_title_to_img.
            face_color(int or tuple of int): default=None, set face color. Refer to add_title_to_img.
            avoid_decorate_axes(bool): default=True, refer to add_title_to_img.

    Returns:
        np.ndarray: gridded image
    """
    # subplot dict
    if sub_plot_dict is None:
        sub_plot_dict = {}

    sub_plot_size_wh = sub_plot_dict.get("size_wh", (128, 128))

    sub_plot_space_tblr = sub_plot_dict.get("space_tblr", (0, 0, 0, 0))
    fill_color = sub_plot_dict.get("fill_color", (255, 255, 255))

    spine_thickness = sub_plot_dict.get("spine_thickness", 0)
    spine_color = sub_plot_dict.get("spine_color", (0, 0, 0))

    # font dict
    if title_font_dict is None:
        title_font_dict = {}

    font_face = title_font_dict.get("font_face", cv2.FONT_HERSHEY_SIMPLEX)
    title_color = title_font_dict.get("color", (0, 0, 0))
    title_face_color = title_font_dict.get("face_color", None)

    title_h_align = title_font_dict.get("horizontal_alignment", "mid")
    title_v_align = title_font_dict.get("vertical_alignment", "center")
    avoid_decorate_axes = title_font_dict.get("avoid_decorate_axes", True)

    # check row, col
    row, col = sub_plot_nrow_ncol
    if np.max(sub_plot_nrow_ncol) <= 0:
        raise ValueError("Row and col cannot be negative at the same time.")
    if row < 0:
        row = math.ceil(len(imgs) / col)
    elif col < 0:
        col = math.ceil(len(imgs) / row)
    if row * col < len(imgs):
        raise ValueError(
            "Row * col mast be equal to or greater than the number of images in 'imgs'."
        )

    # check sub_plot_titles
    if (titles is not None) and (len(titles) != len(imgs)):
        raise ValueError("len(sub_plot_titles) must be equal to len(imgs).")

    # check title thickness
    if title_thickness < 0:
        raise ValueError("'title_thickness' must be a non-negative integer.")

    # check img dimension
    img_dims = [img.ndim for img in imgs]
    min_dim = min(img_dims)
    max_dim = max(img_dims)
    if (min_dim == max_dim == 2) or (min_dim == max_dim == 3):
        cvt_color = False
    elif (min_dim == 2) and (max_dim == 3):
        cvt_color = True
    else:
        raise ValueError("The image in 'imgs' must have 2 or 3 dimensions.")

    # fill color
    if (max_dim == 2) and isinstance(fill_color, tuple):
        fill_color = fill_color[0]
    elif (max_dim == 3) and isinstance(fill_color, int):
        fill_color = (fill_color, 0, 0)

    # make plot_img
    plot_img_list = []
    for img_idx, plot_img in enumerate(imgs):

        # convert color
        if cvt_color and plot_img.ndim == 2:
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_GRAY2BGR)

        # scale
        plot_img = resize_img(
            plot_img, dsize_wh=sub_plot_size_wh, keep_aspect_ratio=True
        )
        plot_img, _ = pad_img(
            plot_img, dsize_wh=sub_plot_size_wh, fill_value=fill_color
        )

        # draw spine
        if spine_thickness > 0:
            plot_img, _ = pad_img(
                plot_img, pad_tblr=(spine_thickness,) * 4, fill_value=spine_color
            )

        # add title
        if titles is not None:
            title = titles[img_idx]
            plot_img = add_title_to_img(
                plot_img,
                title=title,
                color=title_color,
                thickness=title_thickness,
                font_scale=title_scale,
                avoid_decorate_axes=avoid_decorate_axes,
                face_color=title_face_color,
                fill_color=fill_color,
                horizontal_alignment=title_h_align,
                vertical_alignment=title_v_align,
                font_face=font_face,
            )

        # add space between subplots
        plot_img, _ = pad_img(
            plot_img, pad_tblr=sub_plot_space_tblr, fill_value=fill_color
        )

        # add plot_img
        plot_img_list.append(plot_img)

    # make image grid
    fill_img_list = [
        (np.ones(plot_img_list[0].shape) * fill_color).astype(np.uint8)
    ] * (row * col - len(imgs))
    plot_img_list.extend(fill_img_list)

    grid_img_cols = []
    for i in range(row):
        merge_col = np.hstack(plot_img_list[col * i : col * (i + 1)])
        grid_img_cols.append(merge_col)

    grid_img = np.vstack(grid_img_cols)

    return grid_img


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
    if flags == 0:
        flags = cv2.WINDOW_NORMAL
    elif flags == 1:
        flags = cv2.WINDOW_AUTOSIZE
    cv2.namedWindow(win_name, flags)
    cv2.imshow(win_name, img)

    return waitkey(wait_key_ms)
