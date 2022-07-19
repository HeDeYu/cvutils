# -*- coding:utf-8 -*-
# @FileName :core.py
# @Author   :Deyu He
# @Time     :2022/7/19 20:21

# from typing import List, Tuple, Union

import numpy as np

__all__ = [
    "as_array",
    # "check_array",
    "check_array_shape",
]


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
            pass
            # raise_check_array_error(
            #     check_item="shape",
            #     valid_values=valid_shape,
            #     arg_key=array_key,
            #     arg_value=array.shape,
            # )
