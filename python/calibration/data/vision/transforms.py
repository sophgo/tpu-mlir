# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
# coding: utf-8
# pylint: disable= arguments-differ
"Image transforms."

from numbers import Integral
import numpy as np
import cv2

__all__ = [
    "Compose",
    "Normalize",
    "Cast",
    "Resize",
    "CenterCrop",
    "CropResize",
    "Rotate",
]


class Compose(object):
    """Sequentially composes multiple transforms.

    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.


    Inputs:
        - **data**: input tensor with shape of the first transform Block requires.

    Outputs:
        - **out**: output tensor with shape of the last transform Block produces.

    Examples
    --------
    >>> transformer = transforms.Compose([transforms.Resize(300),
    ...                                   transforms.CenterCrop(256),
    ...                                   transforms.ToTensor()])
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <ndArray 3x256x256>
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def __repr__(self):
        def _indent(s_, numSpaces):
            """Indent string"""
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [first] + [(numSpaces * " ") + line for line in s]
            s = "\n".join(s)
            return s

        s = "{name}(\n{modstr}\n)"
        modstr = "\n".join(
            [
                "  ({key}): {block}".format(
                    key=func.__name__, block=_indent(func.__repr__(), 2)
                )
                for func in self.transforms
            ]
        )
        return s.format(name=self.__class__.__name__, modstr=modstr)

    def __getitem__(self, key):
        return self.transforms[key]

    def __len__(self):
        return len(self.transforms)

    def __call__(self, x):
        if len(self) == 0:
            return x
        _x = self[0](x)
        for fun in self.transforms[1:]:
            _x = fun(_x)
        return _x


class Cast:
    """Cast input to a specific data type

    Parameters
    ----------
    dtype : str, default 'float32'
        The target data type, in string or `numpy.dtype`.


    Inputs:
        - **data**: input tensor with arbitrary shape and dtype.

    Outputs:
        - **out**: output tensor with the same shape as `data` and data type as dtype.
    """

    def __init__(self, dtype=np.float32):
        self._dtype = dtype

    def __call__(self, x):
        return x.astype(self._dtype)


class Normalize:
    """Normalize an tensor of shape (H x W x C) with mean and
    standard deviation.

    Given mean `(m1, ..., mn)` and std `(s1, ..., sn)` for `n` channels,
    this transform normalizes each channel of the input tensor with::

        output[i] = (input[i] - mi) / si

    If mean or std is scalar, the same value will be applied to all channels.

    Parameters
    ----------
    mean : float or tuple of floats
        The mean values.
    std : float or tuple of floats
        The standard deviation values.


    Inputs:
        - **data**: input tensor with (H x W x C) or (H x W x C) shape.

    Outputs:
        - **out**: output tensor with the shape as `data`.

    Examples
    --------
    >>> transformer = transforms.Normalize(mean=(0, 1, 2), std=(3, 2, 1))
    >>> image = np.random.uniform(0, 1, (4, 2, 3))
    <NDArray 4x2x3>
    """

    def __init__(self, mean=0.0, std=1.0):
        self._mean = mean
        self._std = std

    def __call__(self, x):
        mean = np.array(self._mean, dtype=x.dtype)
        std = np.array(self._std, dtype=x.dtype)
        return (x - mean) / std


class Rotate:
    """Rotate the input image by a given angle. Keeps the original image shape.

    Parameters
    ----------
    rotation_degrees : float32
        Desired rotation angle in degrees(anti-clockwise).

    Inputs:
        - **data**: input tensor with (H x W x C)

    Outputs:
        - **out**: output tensor with (H x W x c)
    """

    def __init__(self, rotation_degrees):
        self._rotation_degrees = rotation_degrees

    def __call__(self, x):
        # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga744529385e88ef7bc841cbe04b35bfbf
        # https://theailearner.com/tag/cv2-getrotationmatrix2d/
        center = tuple((np.array(x.shape[0:2]) - 1) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, self._rotation_degrees, 1.0)
        return cv2.warpAffine(x, rot_mat, x.shape[0:2], flags=cv2.INTER_LINEAR)


class Resize:
    """Resize an image or a batch of image NDArray to the given size.
    Should be applied before `mxnet.gluon.data.vision.transforms.ToTensor`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    keep_ratio : bool
        Whether to resize the short edge or both edges to `size`,
        if size is give as an integer.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.

    Examples
    --------
    >>> transformer = transforms.Resize(size=(1000, 500))
    >>> image = np.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 500x1000x3>
    """

    def __init__(self, size, keep_ratio=False, interpolation=cv2.INTER_LINEAR):
        self._keep = keep_ratio
        if isinstance(size, Integral):
            size = (size, size)
        self._size = size
        self._interpolation = interpolation

    def __call__(self, x: np.ndarray) -> np.ndarray:

        if self._keep:
            h, w, c = x.shape
            w_n, h_n = self._size
            scale_h, scale_w = h_n / h, w_n / w
            if scale_w == scale_h:
                return cv2.resize(x, self._size, self._interpolation)

            scale = min(scale_h, scale_w)
            W, H = (int(i * scale) for i in (w, h))
            y = cv2.resize(x, (W, H), self._interpolation)
            out = np.zeros((h_n, w_n, c))
            h_c = int(h_n / 2 - H / 2)
            w_c = int(w_n / 2 - W / 2)
            out[h_c : h_c + H, w_c : w_c + W, :] = y
            return out

        return cv2.resize(x, self._size, self._interpolation)


class CropResize:
    r"""Crop the input image with and optionally resize it.

    Makes a crop of the original image then optionally resize it to the specified size.

    Parameters
    ----------
    x : int
        Left boundary of the cropping area
    y : int
        Top boundary of the cropping area
    w : int
        Width of the cropping area
    h : int
        Height of the cropping area
    size : int or tuple of (w, h)
        Optional, resize to new size after cropping
    interpolation : int, optional
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#resize

    Inputs:
        - **data**: input tensor with (H x W x C).

    Outputs:
        - **out**: input tensor with (H x W x C).

    Examples
    --------
    >>> transformer = transforms.CropResize(x=0, y=0, width=100, height=100)
    >>> image = np.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 100x100x3>
    """

    def __init__(self, x, y, width, height, size=None, interpolation=1):
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._size = size
        self._interpolation = interpolation

    def __call__(self, x):
        def crop(x, _x, _y, _w, _h):
            h, w, c = x.shape
            _start = np.array((_y, _x))
            _shape = np.array((_h, _w))
            _end = np.min([_start + _shape, np.array((h, w))], axis=0)
            h, w = _end - _start
            out = np.zeros((_h, _w, c))
            out[:h, :w, :] = x[_y : _y + h, _x : _x + w, :]
            return out

        out = crop(x, self._x, self._y, self._width, self._height)
        if self._size:
            out = Resize(self._size, False, self._interpolation)(out)
        return out


class CenterCrop:
    """Crops the image `src` to the given `size` by trimming on all four
    sides and preserving the center of the image. Upsamples if `src` is
    smaller than `size`.

    Parameters
    ----------
    size : int or tuple of (W, H)
        Size of output image.
    interpolation : int
        Interpolation method for resizing. By default uses bilinear
        interpolation. See OpenCV's resize function for available choices.


    Inputs:
        - **data**: input tensor with (Hi x Wi x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x C) shape.

    Examples
    --------
    >>> transformer = CenterCrop(size=(1000, 500))
    >>> image = nprandom.uniform(0, 255, (2321, 3482, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    <NDArray 500x1000x3>
    """

    def __init__(self, size, interpolation=1):
        if isinstance(size, Integral):
            size = (size, size)
        self._args = (size, interpolation)

    def __call__(self, x):
        size, interpolation = self._args
        h_i, w_i, c = x.shape
        w_o, h_o = size
        if w_o > w_i and h_o > h_i:
            f = Resize((w_o, h_o), True, interpolation)
            return f(x)
        shape_io = np.array([[h_i, w_i], [h_o, w_o]])
        shape_union = np.min(shape_io, axis=0)
        out = np.zeros((h_o, w_o, c))
        shape_range_start = (shape_io - shape_union) / 2
        shape_range_end = shape_range_start + shape_union
        h_i_s, w_i_s, h_o_s, w_o_s = shape_range_start.flatten().astype(int)
        h_i_e, w_i_e, h_o_e, w_o_e = shape_range_end.flatten().astype(int)
        out[h_o_s:h_o_e, w_o_s:w_o_e, :] = x[h_i_s:h_i_e, w_i_s:w_i_e, :]
        return out
