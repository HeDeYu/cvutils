# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/19 20:21

from typing import List, Tuple, Union

import numpy as np
import pyutils

__all__ = [
    "as_array",
    "check_array",
    "check_array_shape",
]


def check_array(
    size: Union[int, List, Tuple] = None,
    ndim: Union[int, List, Tuple] = None,
    shape: Union[int, List, Tuple] = None,
    dtype=None,
    allow_empty: bool = False,
    **kwargs,
):
    """

    For each *key, value* pair in *kwargs*
        - check specified attributes of *value*
        - check that *value* is in *_values*

    Args:
        size(int, list, tuple): number of elements, refer to *valid_value* in function check_attr_in_list
        ndim(int, list, tuple): valid dimensions, refer to *valid_value* in function check_attr_in_list
        shape(int, list, tuple): valid shape, refer to *valid_shape* in check_array_shape
        dtype(scalar data-type, list, tuple): valid data type, refer to *valid_dtype* in check_array_dtype
        allow_empty(bool): empty array allowance
        **kwargs : dict
            *key, value* pairs as keyword arguments to find in *_values*.

    Raises:
        ValueError: If any *value* in *kwargs* is not an object of np.ndarray,
            or the attributes checking failed

        Examples:
        >>> from cvutils import check_array

        >>> check_array(arg=[1, 2, 3])
        TypeError: expect type np.ndarray for `array`

        >>> check_array(allow_empty=True, array=np.array([]))
        >>> check_array(allow_empty=False, array=np.array([]))
        ValueError: expect an non-empty np.ndarray for array

        >>> array_int8 = np.ones((5, 5), dtype=np.int8)
        >>> array_int8_2 = np.ones((5, 5), dtype=np.int8)
        >>> check_array(dtype=np.int8, size=25, array_int8=array_int8)
        >>> check_array(dtype=(np.uint8, np.float32), size=25, array_int8=array_int8, array_int8_2=array_int8)
        ValueError: expect an array with dtype np.uint8 or np.float32,
        while the dtype of array is np.int8

    """

    # check array
    pyutils.check_isinstance((np.ndarray,), **kwargs)

    for key, array in kwargs.items():
        if (not allow_empty) and array.size == 0:
            raise ValueError("expect an non-empty np.ndarray for {!r}".format(key))

    # check size
    if size is not None:
        size = list(size) if isinstance(size, (list, tuple)) else [size]
        pyutils.check_attr_in_list(attr_name="size", valid_values=size, **kwargs)

    # check ndim
    if ndim is not None:
        ndim = list(ndim) if isinstance(ndim, (list, tuple)) else [ndim]
        pyutils.check_attr_in_list(attr_name="ndim", valid_values=ndim, **kwargs)

    # check dtype
    if dtype is not None:
        check_array_dtype(valid_dtype=dtype, **kwargs)

    # check shape
    if shape is not None:
        check_array_shape(valid_shape=shape, **kwargs)


def as_array(array_like, dtype=None, copy_=False, shape=None) -> np.ndarray:
    """Convert an array_like(list, tuple, np.ndarray) object to np.ndarray

    Args:
        array_like(list, tuple, np.ndarray):
        dtype(str or np.dtype): data type of converted array
        copy_(bool): copy or not
        shape: converted array's shape if specified

    Returns:
        np.ndarray: converted np.ndarray object

    Examples:
        >>> from rrcvutils import as_array
        >>> a = [1, 2, 3]
        >>> as_array(a)
        array([1, 2, 3])
        >>> as_array(a, 'float')
        array([1., 2., 3.])
        >>> as_array(a, 'float', shape=[1, 3])
        array([[1., 2., 3.]])
    """
    array = np.array(array_like, dtype, copy=copy_)

    return array if shape is None else array.reshape(shape)


def check_array_shape(valid_shape, **kwargs):
    """

    For each *key, value* pair in *kwargs*, check the shape of *value* is valid

    Args:
        valid_shape(int, list, tuple): target shape of array, use to check whether the shape of the input array:
            - is consistent with 'shape' if 'shape' is a non-nested list or tuple
            - in 'shape' if 'shape' is a list or tuple nested with only one level
        **kwargs : dict
            *key, value* pairs as keyword arguments to find in *_values*.

    Raises:
        ValueError: If any *value*.shape in *kwargs* is not valid

        Examples:
        >>> from rrcvutils import check_array

        >>> check_array(shape=3, array=np.array([1, 2, 3]))
        >>> check_array(shape=[(3, 1), (1, 3)], array=np.array([[1], [2], [3]]))
        >>> check_array(shape=3, array=np.array([[1], [2], [3]]))
        ValueError: expect an array with shape 3, while the ndim of array is (3, 1)

    """

    if not isinstance(valid_shape, (list, tuple)):
        # 3 --> ((3, ), )
        valid_shape = ((valid_shape,),)

    elif all(not isinstance(s, (list, tuple)) for s in valid_shape):
        # (3, 3, 3) --> ((3, 3, 3), )
        valid_shape = (valid_shape,)

    else:
        # (3, (3, 3, 3)) --> ((3, ), (3, 3, 3))
        valid_shape = list(
            map(
                lambda item: tuple(item)
                if isinstance(item, (list, tuple))
                else (item,),
                valid_shape,
            )
        )

    for array_key, array in kwargs.items():

        for shape_item in valid_shape:

            if len(array.shape) == len(shape_item):

                all_same = all(
                    (s1 == -1) or (s1 == s2) for s1, s2 in zip(shape_item, array.shape)
                )
                if all_same:
                    break
        else:
            raise_check_array_error(
                check_item="shape",
                valid_values=valid_shape,
                arg_key=array_key,
                arg_value=array.shape,
            )


def check_array_dtype(valid_dtype, **kwargs):
    """

    For each *key, value* pair in *kwargs*, check the data type of *value* is valid

    Args:

        valid_dtype: data type, use to check whether the data type of the input array:
            - is consistent with or is subdtype of 'dtype' if 'dtype' is numpy data type refefence to
              https://numpy.org.cn/en/reference/arrays/scalars.html
            - in the 'dtype' or is subdtype of the elements if 'dtype' is list or tuple of numpy data type
        **kwargs : dict
            *key, value* pairs as keyword arguments to find in *_values*.


    Raises:
        ValueError: If any *value*.dtype in *kwargs* :
            - is not *valid_dtype*
            - is not subdetype of *valid_dtype*
            - not in *valid_dtype* or there subdtypes

        Examples:
        >>> from cvutils import check_array
        >>> array_int8 = np.ones((5, 5), dtype=np.int8)
        >>> array_int8_2 = np.ones((5, 5), dtype=np.int8)
        >>> check_array_dtype(valid_dtype=np.int8, arg=array_int8)
        >>> check_array_dtype(valid_dtype=np.integer, arg=array_int8)
        >>> check_array_dtype(valid_dtype=(np.uint8, np.float32), array_int8=array_int8, array_int8_2=array_int8_2)
        ValueError: expect an array with dtype np.uint8 or np.float32,
        while the dtype of array is np.int8

    """
    valid_dtype = (
        valid_dtype if isinstance(valid_dtype, (list, tuple)) else (valid_dtype,)
    )

    for dtype in valid_dtype:
        if not np.issctype(dtype):
            raise ValueError(
                f"expect data type of numpy for target data type, "
                f"while {dtype} in valid data type {valid_dtype} is not data type of numpy"
            )

    for array_k, array in kwargs.items():

        data_dtype = array.dtype
        check_res = any(
            np.issubdtype(data_dtype, dtype_item) for dtype_item in valid_dtype
        )

        if not check_res:
            raise_check_array_error(
                check_item="dtype",
                valid_values=valid_dtype,
                arg_key=array_k,
                arg_value=data_dtype,
            )


def raise_check_array_error(check_item, valid_values, arg_key, arg_value):
    valid_values = (
        (valid_values,) if not isinstance(valid_values, (list, tuple)) else valid_values
    )

    item_value_strs = list(map(str, valid_values))
    if len(item_value_strs) > 1:
        valid_str = ", ".join(item_value_strs[:-1]) + " or " + item_value_strs[-1]
    else:
        valid_str = item_value_strs[0]

    raise ValueError(
        f"expect an array with {check_item} {valid_str}, "
        f"while the {check_item} of '{arg_key}' is {arg_value}"
    )
