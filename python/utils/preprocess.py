import os
import PIL
import numpy as np
import cv2
import ast
import argparse
from enum import Enum
from utils.log_setting import setup_logger
from utils.mlir_parser import *
from utils.misc import *
from PIL import Image

logger = setup_logger('root', log_level="INFO")

class YuvType(Enum):
  YUV420_PLANAR = 1
  YUV_NV12 = 2
  YUV_NV21 = 3

supported_customization_format = [
    'RGB_PLANAR',
    'RGB_PACKED',
    'BGR_PLANAR',
    'BGR_PACKED',
    'GRAYSCALE',
    'YUV420_PLANAR',
    'YUV_NV21',
    'YUV_NV12',
    'RGBA_PLANAR',
    'GBRG_RAW',
    'GRBG_RAW',
    'BGGR_RAW',
    'RGGB_RAW',
    ''
]

customization_format_attributes = {
    'RGB_PLANAR':    ('rgb', 'nchw'),
    'RGB_PACKED':    ('rgb', 'nhwc'),
    'BGR_PLANAR':    ('bgr', 'nchw'),
    'BGR_PACKED':    ('bgr', 'nhwc'),
    'GRAYSCALE':     ('gray', 'nchw'),
    'YUV420_PLANAR': ('bgr', 'nchw'),
    'YUV_NV12':      ('bgr', 'nchw'),
    'YUV_NV21':      ('bgr', 'nchw'),
    'RGBA_PLANAR':   ('rgba', 'nchw'),
    'GBRG_RAW':      ('gbrg', 'nchw'),
    'GRBG_RAW':      ('grbg', 'nchw'),
    'BGGR_RAW':      ('bggr', 'nchw'),
    'RGGB_RAW':      ('rggb', 'nchw')
}

# fix bool bug of argparse


class ImageResizeTool:
    @staticmethod
    def stretch_resize(image, h, w, use_pil_resize=False):
        if use_pil_resize:
            image = image.resize((w, h), PIL.Image.BILINEAR)
            return np.array(image)
        else:
            return cv2.resize(image, (w, h))  # w,h

    @staticmethod
    def letterbox_resize(image, h, w, pad_value=0, pad_type='center', use_pil_resize=False):
        if use_pil_resize:
            iw, ih = image.size
        else:
            ih = image.shape[0]
            iw = image.shape[1]
        scale = min(float(w) / iw, float(h) / ih)
        rescale_w = int(iw * scale)
        rescale_h = int(ih * scale)
        if use_pil_resize:
            resized_img = image.resize(
                (rescale_w, rescale_h), PIL.Image.BILINEAR)
            resized_img = np.array(resized_img)
        else:
            resized_img = cv2.resize(image, (rescale_w, rescale_h))
        paste_w = 0
        paste_h = 0
        if pad_type == 'center':
            paste_w = (w - rescale_w) // 2
            paste_h = (h - rescale_h) // 2
        if resized_img.ndim == 3 and resized_img.shape[2] == 3:
            new_image = np.full((h, w, 3), pad_value, dtype=resized_img.dtype)
            new_image[paste_h:paste_h + rescale_h,
                      paste_w: paste_w + rescale_w, :] = resized_img
            return new_image
        elif resized_img.ndim == 2:
            new_image = np.full((h, w), pad_value, dtype=resized_img.dtype)
            new_image[paste_h:paste_h + rescale_h,
                      paste_w: paste_w + rescale_w] = resized_img
            return new_image
        raise RuntimeError("invalid image shape:{}".format(resized_img.shape))

    @staticmethod
    def short_side_scale_resize(image, h, w, use_pil_resize=False):
        if use_pil_resize:
            iw, ih = image.size
        else:
            ih = image.shape[0]
            iw = image.shape[1]
        scale = max(float(w) / iw, float(h) / ih)
        rescale_w = int(iw * scale) if iw * scale >= w else w
        rescale_h = int(ih * scale) if ih * scale >= h else h
        if use_pil_resize:
            resized_img = image.resize(
                (rescale_w, rescale_h), PIL.Image.BILINEAR)
            resized_img = np.array(resized_img)
        else:
            resized_img = cv2.resize(image, (rescale_w, rescale_h))
        #center_crop to make sure resized_img shape is (h,w)
        start_h = (rescale_h - h) // 2 if rescale_h - h > 0 else 0
        start_w = (rescale_w - w) // 2 if rescale_w - w > 0 else 0
        resized_img = resized_img[start_h : start_h + h, start_w : start_w + w]
        return resized_img


def add_preprocess_parser(parser):
    parser.add_argument("--resize_dims", type=str,
                        help="Image was resize to fixed 'h,w', default is same as net input dims")
    parser.add_argument("--keep_aspect_ratio", action='store_true', default=False,
                        help="Resize image by keeping same ratio, any areas which" +
                             "are not taken are filled with 0")
    parser.add_argument("--keep_ratio_mode", choices=['letterbox', 'short_side_scale'], default='letterbox',
                        help = "If use keep_aspect_ratio, different mode for resize")
    parser.add_argument("--mean", default='0,0,0', nargs='?',
                        help="Per Channel image mean values")
    parser.add_argument("--scale", default='1,1,1',
                        help="Per Channel image scale values")
    parser.add_argument("--pixel_format", choices=['rgb', 'bgr', 'gray', 'rgba', 'gbrg', 'grbg', 'bggr', 'rggb' ], default='bgr',
                        help='fixel format of output data that sent into model')
    parser.add_argument("--channel_format", choices=['nhwc', 'nchw', 'none'], default='nchw',
                        help='channel first or channel last, or not image')
    parser.add_argument("--pad_value", type=int, default=0,
                        help="pad value when resize ")
    parser.add_argument("--pad_type", type=str, choices=[
                        'normal', 'center'], default='center', help="type of pad when resize ")
    parser.add_argument("--preprocess_list", type=str2list, default=list(),
                        help = "choose which input need preprocess, like:'1,3' means input 1&3 need preprocess, default all inputs")
    parser.add_argument("--debug_cmd", type=str, default='', help="debug cmd")
    avoid_opts = parser.add_argument_group('avoid options')
    avoid_opts.add_argument('unknown_params', nargs='?', default=[], help='not parameters but starting with "-"')

    return parser


def get_preprocess_parser(existed_parser=None):
    if existed_parser:
        if not isinstance(existed_parser, argparse.ArgumentParser):
            raise RuntimeError("parser is invaild")
        parser = existed_parser
    else:
        parser = argparse.ArgumentParser(description="Image Preprocess.")
    return add_preprocess_parser(parser)


class preprocess(object):
    def __init__(self, debug_cmd=''):
        self.debug_cmd = debug_cmd
        self.fuse_pre = False
        self.has_pre = False
        pass

    def config(self, resize_dims=None, keep_aspect_ratio=False, keep_ratio_mode = "letterbox",
               customization_format = None, fuse_pre = False, aligned = False,
               mean='0,0,0', scale='1,1,1', pixel_format='bgr', pad_type='center', pad_value=0, chip = "",
               channel_format='nchw', preprocess_list: list = [], debug_cmd='', input_shapes=None, unknown_params=[], **ignored):  # add input_shapes for model_eval.py by wangxuechuan 20221110
        if self.debug_cmd == '':
            self.debug_cmd = debug_cmd
        if preprocess_list is not None and preprocess_list != []:
            self.preprocess_list = [ int(i) for i in preprocess_list ]
        else:
            self.preprocess_list = None
        if input_shapes is not None and input_shapes != [] and channel_format != 'none':
            if isinstance(input_shapes, str):
                input_shapes = str2shape(input_shapes)
            self.batch_size = input_shapes[0][0]
            self.net_input_dims = input_shapes[0][-2:]
            if channel_format == 'nhwc':
                if len(input_shapes[0]) >= 4:
                    self.net_input_dims = input_shapes[0][-3:-1]
                else:
                    self.net_input_dims = input_shapes[0][:-1]
        else:
            self.net_input_dims = None
        if resize_dims:
            if isinstance(resize_dims, str):
                self.resize_dims = [int(s) for s in resize_dims.split(',')]
            elif isinstance(resize_dims, list):
                self.resize_dims = resize_dims
            else:
                assert("resize_dims should either be str or list.")
        else:
            self.resize_dims = self.net_input_dims
        self.crop_method = 'center'
        self.keep_aspect_ratio = keep_aspect_ratio
        self.keep_ratio_mode = keep_ratio_mode
        self.pad_value = pad_value
        self.pad_type = pad_type
        self.pixel_format = pixel_format
        self.channel_format = channel_format

        self.input_name = 'input'
        self.channel_num = 3
        if self.pixel_format == 'gray':
            self.channel_num = 1
        elif self.pixel_format == 'rgba':
            self.channel_num = 4

        if unknown_params:
            self.mean = np.array([float(s) for sublist in unknown_params for s in sublist.split(',')], dtype=np.float32)
        else:
            self.mean = np.array([float(s)
                             for s in mean.split(',')], dtype=np.float32)
        self.mean = self.mean[np.newaxis, :, np.newaxis, np.newaxis]
        assert (self.mean.size >= self.channel_num)
        self.scale = np.array([float(s)
                              for s in scale.split(',')], dtype=np.float32)
        self.scale = self.scale[np.newaxis, :, np.newaxis, np.newaxis]
        assert (self.scale.size >= self.channel_num)

        self.aligned = aligned
        self.customization_format = customization_format
        self.fuse_pre = fuse_pre
        self.has_pre = True
        #fuse_preprocess for cv18xx
        if self.fuse_pre:
            #resize_dims should be greater than net_input_dims
            self.resize_dims = [max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dims)]
            self.net_input_dims = self.resize_dims
            self.pixel_format = customization_format_attributes[self.customization_format][0]
            self.channel_format = customization_format_attributes[self.customization_format][1]
            if self.customization_format.find("YUV") >= 0:
                self.aligned = True
            #aligned_input for cv18xx
            if str(chip).lower().endswith('183x'):
                self.VPSS_W_ALIGN = 32
                self.VPSS_Y_ALIGN = 32
                self.VPSS_CHANNEL_ALIGN = 4096
                if self.customization_format == "YUV420_PLANAR":
                    self.VPSS_Y_ALIGN = self.VPSS_W_ALIGN * 2
            else:
               self.VPSS_W_ALIGN = 64
               self.VPSS_Y_ALIGN = 64
               self.VPSS_CHANNEL_ALIGN = 64
               if self.customization_format == "YUV420_PLANAR":
                   self.VPSS_Y_ALIGN = self.VPSS_W_ALIGN * 2

        info_str = \
            "\n\t _____________________________________________________ \n" + \
            "\t| preprocess:                                           |\n" + \
            "\t|   (x - mean) * scale                                  |\n" + \
            "\t'-------------------------------------------------------'\n"

        format_str = "  config Preprocess args : \n" + \
            "\tresize_dims           : {}\n" + \
            "\tkeep_aspect_ratio     : {}\n" + \
            "\tkeep_ratio_mode       : {}\n" + \
            "\tpad_value             : {}\n" + \
            "\tpad_type              : {}\n" + \
            "\t--------------------------\n" + \
            "\tmean                  : {}\n" + \
            "\tscale                 : {}\n" + \
            "\t--------------------------\n" + \
            "\tpixel_format          : {}\n" + \
            "\tchannel_format        : {}\n"
        resize_dims_str = resize_dims if resize_dims is not None else 'same to net input dims'
        info_str += format_str.format(resize_dims_str, self.keep_aspect_ratio, self.keep_ratio_mode, self.pad_value, self.pad_type,
                                      list(self.mean.flatten()), list(
                                          self.scale.flatten()), self.pixel_format, self.channel_format)
        logger.info(info_str)

    def load_config(self, input_op):
        self.input_name = Operation.name(input_op)
        shape = Operation.shape(input_op)
        self.net_input_dims = []
        if len(shape) >= 3:
            self.net_input_dims = shape[-2:]
            self.batch_size = shape[0]
        elif len(shape) == 2:
            #in some model, the input tensor's dims maybe 2
            self.net_input_dims = shape[0:]
            self.batch_size = shape[0]
        else:
            return

        # ignore those non preprocess attrs
        attrs = input_op.attributes
        non_preprc_attr = ["is_shape", "do_preprocess", "customization_format", "aligned"]
        for attr in non_preprc_attr:
            if attr in attrs:
                del attrs[attr]

        if len(attrs) <= 1:
            return

        self.pixel_format = Operation.str(attrs['pixel_format'])
        self.channel_num = 3
        if self.pixel_format == 'gray':
            self.channel_num = 1
        elif self.pixel_format == 'rgba':
            self.channel_num = 4
        self.channel_format = Operation.str(attrs['channel_format'])
        if self.channel_format == 'nhwc':
            self.net_input_dims = shape[1:-1]
        self.keep_aspect_ratio = Operation.bool(attrs['keep_aspect_ratio'])
        self.keep_ratio_mode = Operation.str(attrs['keep_ratio_mode'])
        self.pad_value = Operation.int(attrs['pad_value'])
        self.pad_type = Operation.str(attrs['pad_type'])
        try:
            self.resize_dims = Operation.int_array(attrs['resize_dims'])
        except KeyError:
            self.resize_dims = self.net_input_dims
        if len(self.resize_dims) == 0 or self.resize_dims is None:
            self.resize_dims = self.net_input_dims
        self.mean = np.array(Operation.fp_array(
            attrs['mean'])).astype(np.float32)
        self.mean = self.mean[np.newaxis, :, np.newaxis, np.newaxis]
        self.scale = np.array(Operation.fp_array(
            attrs['scale'])).astype(np.float32)
        self.scale = self.scale[np.newaxis, :, np.newaxis, np.newaxis]
        self.crop_method = 'center'
        self.has_pre = True
        format_str = "\n  load_config Preprocess args : \n" + \
            "\tresize_dims           : {}\n" + \
            "\tkeep_aspect_ratio     : {}\n" + \
            "\tkeep_ratio_mode       : {}\n" + \
            "\tpad_value             : {}\n" + \
            "\tpad_type              : {}\n" + \
            "\tinput_dims            : {}\n" + \
            "\t--------------------------\n" + \
            "\tmean                  : {}\n" + \
            "\tscale                 : {}\n" + \
            "\t--------------------------\n" + \
            "\tpixel_format          : {}\n" + \
            "\tchannel_format        : {}\n"
        logger.info(format_str.format(self.resize_dims, self.keep_aspect_ratio, self.keep_ratio_mode, self.pad_value, self.pad_type,
                                      self.net_input_dims,
                                      list(self.mean.flatten()), list(
                                          self.scale.flatten()),
                                      self.pixel_format,
                                      self.channel_format))

    def to_dict(self):
        if not self.has_pre:
            return {}
        return {
            'preprocess_list':self.preprocess_list,
            'resize_dims': self.resize_dims,
            'keep_aspect_ratio': self.keep_aspect_ratio,
            'keep_ratio_mode': self.keep_ratio_mode,
            'pad_value': self.pad_value,
            'pad_type': self.pad_type,
            'mean': list(self.mean.flatten()),
            'scale': list(self.scale.flatten()),
            'pixel_format': self.pixel_format,
            'channel_format': self.channel_format
        }

    def __right_crop(self, img, crop_dim):
        ih, iw = img.shape[2:]
        oh, ow = crop_dim
        img = img[:, :, ih-oh:, iw-ow:]
        return img

    def __center_crop(self, img, crop_dim):
        # Take center crop.
        h, w = img.shape[2:]
        crop_h, crop_w = crop_dim
        start_h = (h // 2) - (crop_h // 2)
        start_w = (w // 2) - (crop_w // 2)
        img = img[:, :, start_h: (start_h + crop_h),
                  start_w: (start_w + crop_w)]
        return img

    def __load_image_and_resize(self, input):
        image = None
        image_path = str(input).rstrip()
        if not os.path.exists(image_path):
            print("{} doesn't existed !!!".format(image_path))
            exit(1)

        use_pil_resize = False
        if 'use_pil_resize' in self.debug_cmd:
            use_pil_resize = True
            if self.channel_num == 1:
                image = Image.open(image_path).convert("L")
            elif self.channel_num == 3:
                image = Image.open(image_path).convert("RGB")
                # convert from RGB to BGR
                image = Image.fromarray(np.array(image)[:, :, [2, 1, 0]])
            elif self.channel_num == 4:
                image = PIL.Image.open(image_path).convert('RGBA')
            width, height = image.size
            ratio = min(self.net_input_dims[0] /
                        height, self.net_input_dims[1] / width)
        else:
            if self.channel_num == 1:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            elif self.channel_num == 3:
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            elif self.channel_num == 4:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image.shape[-1] != 4:
                    image = PIL.Image.open(image_path).convert('RGBA')
                    image = np.array(image)
                else:
                    # convert from BGRA to RGBA
                    image = image[:, :, [2, 1, 0, 3]]
            ratio = min(
                self.net_input_dims[0] / image.shape[0], self.net_input_dims[1] / image.shape[1])
        if self.keep_aspect_ratio:
            if self.keep_ratio_mode == "letterbox":
                image = ImageResizeTool.letterbox_resize(
                    image, self.resize_dims[0], self.resize_dims[1], self.pad_value, self.pad_type, use_pil_resize)
            else:
                image = ImageResizeTool.short_side_scale_resize(
                    image, self.resize_dims[0], self.resize_dims[1], use_pil_resize)
        else:
            image = ImageResizeTool.stretch_resize(
                image, self.resize_dims[0], self.resize_dims[1], use_pil_resize)

        if self.channel_num == 1:
            # if grapscale image, expand dim to (1, h, w)
            image = np.expand_dims(image, axis=0)
        else:
            # opencv read image data format is hwc, tranpose to chw
            image = np.transpose(image, (2, 0, 1))
        return image, ratio

    def get_config(self, attr_type):
        if attr_type == 'ratio':
            return self.ratio_list

    def align_up(self, x, n):
        return x if n == 0 else ((x + n - 1)// n) * n

    # Y = 0.2569 * R + 0.5044 * G + 0.0979 * B + 16
    # U = -0.1483 * R - 0.2911 * G + 0.4394 * B + 128
    # V = 0.4394 * R - 0.3679 * G - 0.0715 * B + 128
    def rgb2yuv420(self, input, pixel_type):
        # every 4 y has one u,v
        # vpss format, w align is 32, channel align is 4096
        h, w, c = input.shape
        y_w_aligned = self.align_up(w, self.VPSS_Y_ALIGN)
        y_offset = 0
        if pixel_type == YuvType.YUV420_PLANAR:
          uv_w_aligned = self.align_up(int(w/2), self.VPSS_W_ALIGN)
          u_offset = self.align_up(y_offset + h * y_w_aligned, self.VPSS_CHANNEL_ALIGN)
          v_offset = self.align_up(u_offset + int(h/2) * uv_w_aligned, self.VPSS_CHANNEL_ALIGN)
        else :
          uv_w_aligned = self.align_up(w, self.VPSS_W_ALIGN)
          u_offset = self.align_up(y_offset + h * y_w_aligned, self.VPSS_CHANNEL_ALIGN)
          v_offset = u_offset

        total_size = self.align_up(v_offset + int(h/2) * uv_w_aligned, self.VPSS_CHANNEL_ALIGN)
        yuv420 = np.zeros(int(total_size), np.uint8)
        for h_idx in range(h):
            for w_idx in range(w):
                r, g, b = input[h_idx][w_idx]
                y = int(0.2569 * r + 0.5044 * g + 0.0979 * b + 16)
                u = int(-0.1483 * r - 0.2911 * g + 0.4394 * b + 128)
                v = int(0.4394 * r - 0.3679 * g - 0.0715 * b + 128)
                y = max(min(y, 255), 0)
                u = max(min(u, 255), 0)
                v = max(min(v, 255), 0)
                yuv420[y_offset + h_idx * y_w_aligned + w_idx] = y
                if (h_idx % 2 == 0 and w_idx % 2 == 0):
                  if pixel_type == YuvType.YUV420_PLANAR:
                    u_idx = int(u_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2))
                    v_idx = int(v_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2))
                  elif pixel_type == YuvType.YUV_NV12:
                    u_idx = int(u_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2)
                    v_idx = int(v_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2 + 1)
                  else :
                    u_idx = int(u_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2 + 1)
                    v_idx = int(v_offset + int(h_idx/2) * uv_w_aligned + int(w_idx / 2) * 2)

                  yuv420[u_idx] = u
                  yuv420[v_idx] = v
        return yuv420.reshape(int(total_size), 1, 1)

    def align_packed_frame(self, x, aligned):
        if not aligned:
            return x
        h, w, c = x.shape
        w = w * c
        x = np.reshape(x, (1, h, w))
        x_tmp = np.zeros((1, h, self.align_up(w, self.VPSS_W_ALIGN)), x.dtype)
        x_tmp[:, :, : w] = x
        return x_tmp

    def align_gray_frame(self, x, aligned):
        if not aligned:
            return x
        c, h, w = x.shape
        x_tmp = np.zeros((c, h, self.align_up(w, self.VPSS_W_ALIGN)), x.dtype)
        x_tmp[:, :, :w] = x
        return x_tmp

    def align_planar_frame(self, x, aligned):
        if not aligned:
          return x
        c, h, w = x.shape
        align_w_size = self.align_up(w, self.VPSS_W_ALIGN)
        align_c_size = self.align_up(align_w_size * h, self.VPSS_CHANNEL_ALIGN)
        x_tmp1 = np.zeros((c, h, align_w_size), x.dtype)
        x_tmp1[:, :, :w] = x
        x_tmp1 = np.reshape(x_tmp1, (c, 1, h * align_w_size))
        x_tmp2 = np.zeros((c, 1, align_c_size), x.dtype)
        x_tmp2[:, :, : h * align_w_size] = x_tmp1
        return x_tmp2

    def run(self, input):
        # load and resize image, the output image is chw format.
        x_list = []
        self.ratio_list = []
        for path in input.split(','):
            x, ratio = self.__load_image_and_resize(path)
            x = np.expand_dims(x, axis=0)
            x_list.append(x)
            self.ratio_list.append(ratio)
        x = np.concatenate(x_list, axis=0)
        # take center crop if needed
        if self.resize_dims != self.net_input_dims:
            if self.crop_method == "right":
                x = self.__right_crop(x, self.net_input_dims)
            else:
                x = self.__center_crop(x, self.net_input_dims)

        x = x.astype(np.float32)
        if self.fuse_pre:
            x = np.squeeze(x, 0)
            if self.customization_format == "GRAYSCALE":
                x = self.align_gray_frame(x, self.aligned)
                x = np.expand_dims(x, axis=0)
                x = x.astype(np.uint8)
            elif self.customization_format.find("YUV") >= 0:
                # swap to 'rgb'
                pixel_type = YuvType.YUV420_PLANAR
                if self.customization_format == 'YUV420_PLANAR':
                    pixel_type = YuvType.YUV420_PLANAR
                elif self.customization_format == 'YUV_NV12':
                    pixel_type = YuvType.YUV_NV12
                else:
                    pixel_type = YuvType.YUV_NV21
                x = x[[2, 1, 0], :, :]
                x = np.transpose(x, (1, 2, 0))
                x = self.rgb2yuv420(x, pixel_type)
                x = x.astype(np.uint8)
                assert(self.batch_size == 1)
            elif self.customization_format.find("_PLANAR") >= 0:
                if self.pixel_format == 'rgb':
                    x = x[[2, 1, 0], :, :]
                x = self.align_planar_frame(x, self.aligned)
                x = np.expand_dims(x, axis=0)
                x = x.astype(np.uint8)
            elif self.customization_format.find("_PACKED") >= 0:
                if self.pixel_format == 'rgb':
                    x = x[[2, 1, 0], :, :]
                x = np.transpose(x, (1, 2, 0))
                x = self.align_packed_frame(x, self.aligned)
                x = np.expand_dims(x, axis=0)
                x = x.astype(np.uint8)
            else:
                logger.info("unsupported pixel format");
                assert(0)
        else:
            if self.pixel_format == 'gray':
                self.mean = self.mean[:, :1, :, :]
                self.scale = self.scale[:, :1, :, :]
            elif self.pixel_format == 'rgb':
                x = x[:, [2, 1, 0], :, :]
            x = (x - self.mean) * self.scale

            if self.channel_format == 'nhwc':
                x = np.transpose(x, (0, 2, 3, 1))

        if len(input.split(',')) == 1:
            x = np.repeat(x, self.batch_size, axis=0)
        return x
