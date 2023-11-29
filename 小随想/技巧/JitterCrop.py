import cv2
import numpy as np


def present(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


def cropper(image, crop_shape, jitter=False, max_jitter=None, planes=None):
    """
    Crops 3d data
    :param image: 3d array, image to be cropped
    :param crop_shape: tuple, crop shape
    :param jitter: boolean, randomly move the center point within a given limit, default is False
    :param max_jitter: tuple, maximum displacement for jitter, if None then it gets a default value
    :param planes: scalar, get a crop image plane
    :return: 3d array
    """

    half_crop_shape = tuple(_c // 2 for _c in crop_shape)
    half_image_shape = tuple(_i // 2 for _i in image.shape)
    assert all(
        [_c <= _i for _c, _i in zip(half_crop_shape, half_image_shape)]
    ), "Crop shape is bigger than image shape"

    if jitter:
        contrast_1 = tuple(
            (_i - _c) // 4 for _c, _i in zip(half_crop_shape, half_image_shape)
        )
        contrast_2 = tuple(c // 2 for c in half_crop_shape)
        if max_jitter is None:
            max_jitter = tuple(
                [min(_ct2, _ct1) for _ct2, _ct1 in zip(contrast_1, contrast_2)]
            )
        all(
            [
                _i - _m >= 0 and _i + _m < 2 * _i
                for _m, _i in zip(max_jitter, half_image_shape)
            ]
        ) or present(
            ValueError(
                "Jitter results in cropping outside border, please reduce max_jitter"
            )
        )
        loc = tuple(
            _l - np.random.randint(-1 * max_jitter[_i], max_jitter[_i])
            for _i, _l in enumerate(half_image_shape)
        )
    else:
        loc = half_image_shape

    crop_image = image[
        loc[0] - half_crop_shape[0] : loc[0] + half_crop_shape[0],
        loc[1] - half_crop_shape[1] : loc[1] + half_crop_shape[1],
        loc[2] - half_crop_shape[2] : loc[2] + half_crop_shape[2],
    ]

    if planes is not None:
        try:
            crop_image = crop_image[planes]
        except IndexError:
            present(ValueError("Plane does not exist"))

    return crop_image


def count_place(data, path):
    data_gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

    y_sum = np.sum(data_gray, axis=1)
    x_sum = np.sum(data_gray, axis=0)
    up = 0
    down = data.shape[0]
    left = 0
    right = data.shape[1]

    try:
        up = (
            min(np.where(y_sum != 0)[0]) - 20
            if min(np.where(y_sum != 0)[0]) > 20
            else 0
        )
        down = (
            max(np.where(y_sum != 0)[0]) + 20
            if max(np.where(y_sum != 0)[0]) < data.shape[0] - 20
            else data.shape[0]
        )
        left = (
            min(np.where(x_sum != 0)[0]) - 20
            if min(np.where(x_sum != 0)[0]) > 20
            else 0
        )
        right = (
            max(np.where(x_sum != 0)[0]) + 20
            if min(np.where(x_sum != 0)[0]) < data.shape[1] - 20
            else data.shape[1]
        )
        error_flag = 0
    except ValueError:
        error_flag = 1
        print(path)

    return up, down, left, right, error_flag
