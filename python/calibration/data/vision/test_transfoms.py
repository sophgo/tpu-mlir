# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

from .transforms import *
import numpy as np


def test_Compose():
    import math

    comp = Compose([math.sqrt, math.exp])
    comp(2)
    print(comp(2))
    assert comp[0](9) == 3
    assert len(comp) == 2


class TestCast:

    def test_Cast_i32(self):
        i32 = Cast(np.uint32)
        x = np.arange(10, dtype=np.uint8)
        y = i32(x)
        assert y.dtype == np.uint32
        assert all(y == x)

    def test_Cast_f32(self):
        i32 = Cast(np.float32)
        x = np.arange(10, dtype=np.int8)
        y = i32(x)
        assert y.dtype == np.float32
        assert (y == x).all()


class TestNormalize:
    x = np.random.uniform(0, 1, (4, 2, 3))

    def test_Normalize(self):
        norm = Normalize(mean=(0, 1, 2), std=(3, 2, 1))

        y = norm(self.x)
        assert y.shape == self.x.shape

    def test_Normalize_scalar(self):
        norm = Normalize(mean=(0, 1, 2), std=3)
        y = norm(self.x)

    def test_Normalize_scalar_(self):
        norm = Normalize(mean=(0, 1, 2), std=(3))
        y = norm(self.x)


def test_Rotate():
    rot = Rotate(90)
    x = np.random.uniform(0, 255, (6, 6, 3)).astype(np.uint8)
    y = rot(x)
    assert (x == y[::-1, :, :].transpose((1, 0, 2))).all()


class TestResize:
    x = np.random.uniform(0, 255, (20, 20, 3)).astype(dtype=np.uint8)

    def test_Resize(self):
        resize = Resize(size=(10, 20))
        y = resize(self.x)
        assert y.shape == (20, 10, 3)
        assert (y[0:5, :, :] != 0).any()

    def test_Resize_keepRatio(self):

        resize = Resize(size=(10, 20), keep_ratio=True)
        y = resize(self.x)
        assert y.shape == (20, 10, 3)
        assert (y[0:5, :, :] == 0).all()
        assert (y[-5:, :, :] == 0).all()


class TestCropResize:
    x = np.random.uniform(0, 255, (20, 20, 3)).astype(dtype=np.uint8)

    def test_CropResize_crop(self):
        cr = CropResize(x=0, y=0, width=10, height=6)
        y = cr(self.x)
        assert y.shape == (6, 10, 3)
        assert (y == self.x[:6, :10, :]).all()

    def test_CropResize_resize(self):
        cr = CropResize(x=0, y=0, width=10, height=6, size=(10, 6))
        y = cr(self.x)
        assert y.shape == (6, 10, 3)
        assert (y == self.x[:6, :10, :]).all()

    def test_CropResize_offset(self):
        cr = CropResize(x=15, y=15, width=10, height=6)
        y = cr(self.x)
        assert y.shape == (6, 10, 3)
        assert (y[:5, :5, :] == self.x[15:, 15:, :]).all()
        z = y.copy()
        z[:5, :5, :] = 0
        assert (z == 0).all()

    def test_CropResize_offset_resize(self):
        cr = CropResize(x=15, y=15, width=10, height=6, size=(20, 20))
        y = cr(self.x)
        assert y.shape == (20, 20, 3)


class TestCenterCrop:
    x = np.random.uniform(0, 255, (20, 20, 3)).astype(dtype=np.uint8)

    def test_CenterCrop(self):
        cc = CenterCrop(size=(10, 7))
        y = cc(self.x)
        assert y.shape == (7, 10, 3)
        assert (y == self.x[6:13, 5:15:, :]).all()

    def test_CenterCrop_resize(self):
        cc = CenterCrop(size=(30, 27))
        y = cc(self.x)
        assert y.shape == (27, 30, 3)


class TestAll:

    def test_resnet_preProcess(self):
        # https://github.com/onnx/models/blob/main/vision/classification/imagenet_preprocess.py
        from . import transforms

        transform_fn = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Cast(np.float32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = np.random.uniform(0, 255, (320, 486, 3)).astype(dtype=np.uint8)
        img = transform_fn(img)
        img = np.expand_dims(img, axis=0)


class TestDataLoader:

    def test_dataset(self):
        from .datasets import ImageFolderDataset
        from . import transforms
        import numpy as np

        imgset = ImageFolderDataset("../../../../regression/image")

        transform_fn = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Cast(np.float32),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            lambda img: np.expand_dims(img, axis=0),
        ])

        img = transform_fn(imgset[0])
        assert img.shape == (1, 224, 224, 3)
        np.savez("resnet50_input.npz", **{"input_1": img})
