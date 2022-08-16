import os
import PIL
import numpy as np
import cv2
import ast
import argparse
from utils.log_setting import setup_logger
from utils.mlir_parser import *

logger = setup_logger('root', log_level="INFO")

# fix bool bug of argparse
def str2bool(v):
  return v.lower() in ("yes", "true", "1")

class ImageResizeTool:
    @staticmethod
    def stretch_resize(image, h, w):
        return cv2.resize(image, (w, h)) # w,h

    @staticmethod
    def letterbox_resize(image, h, w, pad_value = 0, pad_type = 'center'):
        ih = image.shape[0]
        iw = image.shape[1]
        scale = min(float(w) / iw, float(h) / ih)
        rescale_w = int(iw * scale)
        rescale_h = int(ih * scale)
        resized_img = cv2.resize(image, (rescale_w, rescale_h))
        if pad_type == 'center':
            paste_w = (w - rescale_w) // 2
            paste_h = (h - rescale_h) // 2
        if image.ndim == 3 and image.shape[2] == 3:
            new_image = np.full((h, w, 3), pad_value, dtype=image.dtype)
            if pad_type == 'center':
                new_image[paste_h:paste_h + rescale_h,
                          paste_w: paste_w + rescale_w, :] = resized_img
            else:
                new_image[: rescale_h,
                        : rescale_w, :] = resized_img
            return new_image
        elif image.ndim == 2:
            new_image = np.full((h, w),pad_value, dtype=image.dtype)
            if pad_type == 'center':
                new_image[paste_h:paste_h + rescale_h,
                        paste_w: paste_w + rescale_w] = resized_img
            else:
                new_image[:rescale_h,
                        : rescale_w] = resized_img
            return new_image
        raise RuntimeError("invalid image shape:{}".format(image.shape))

def add_preprocess_parser(parser):
    parser.add_argument("--net_input_dims", type=str,
                         help="model's input heigh/width dimension")
    parser.add_argument("--resize_dims", type=str,
                        help="Image was resize to fixed 'h,w', default is same as net input dims")
    parser.add_argument("--keep_aspect_ratio", action='store_true', default=False,
                        help="Resize image by keeping same ratio, any areas which" +
                             "are not taken are filled with 0")
    parser.add_argument("--mean", default='0,0,0', help="Per Channel image mean values")
    parser.add_argument("--scale", default='1,1,1', help="Per Channel image scale values")
    parser.add_argument("--pixel_format", choices=['rgb','bgr','gray','rgba'], default='bgr',
                        help='fixel format of output data that sent into model')
    parser.add_argument("--channel_format", choices=['nhwc','nchw'], default='nchw',
                        help='channel first or channel last')
    parser.add_argument("--pad_value", type=int, default=0, help="pad value when resize ")
    parser.add_argument("--pad_type", type=str, default='center', help="type of pad when resize ")
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
    def __init__(self):
        pass

    def config(self, net_input_dims=None, resize_dims=None, keep_aspect_ratio=False,
               mean='0,0,0', scale='1,1,1', pixel_format='bgr', pad_type='center', pad_value=0,
               channel_format='nchw', **ignored):
        self.batch_size = 1
        if net_input_dims:
            input_shapes = ast.literal_eval(net_input_dims)
            self.net_input_dims = input_shapes[0][-2:]
            self.batch_size = input_shapes[0][0]
            if not resize_dims:
                self.resize_dims = self.net_input_dims
        if resize_dims:
            self.resize_dims = [int(s) for s in resize_dims.split(',')]
            if not net_input_dims:
                self.net_input_dims = self.resize_dims
            self.resize_dims = [max(x,y) for (x,y) in zip(self.resize_dims, self.net_input_dims)]
        if not net_input_dims and not resize_dims:
            self.net_input_dims=[]
            self.resize_dims=[]

        self.crop_method = 'center'
        self.keep_aspect_ratio = keep_aspect_ratio
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

        self.mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
        self.mean = self.mean[np.newaxis, :,np.newaxis, np.newaxis]
        assert(self.mean.size >= self.channel_num)
        self.scale = np.array([float(s) for s in scale.split(',')], dtype=np.float32)
        self.scale = self.scale[np.newaxis, :,np.newaxis, np.newaxis]
        assert(self.scale.size >= self.channel_num)
        if self.channel_format == 'nhwc':
            self.mean = self.mean.transpose(0, 2, 3, 1)
            self.scale = self.scale.transpose(0, 2, 3, 1)


        info_str = \
            "\n\t _____________________________________________________ \n" + \
            "\t| preprocess:                                           |\n" + \
            "\t|   (x - mean) * scale                                  |\n" + \
            "\t'-------------------------------------------------------'\n"

        format_str = "  config Preprocess args : \n" + \
               "\tresize_dims           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\tpad_value             : {}\n" + \
               "\tpad_type              : {}\n" + \
               "\t--------------------------\n" + \
               "\tmean                  : {}\n" + \
               "\tscale                 : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n" + \
               "\tchannel_format        : {}\n"
        resize_dims_str = resize_dims if resize_dims is not None else 'same to net input dims'
        info_str += format_str.format(resize_dims_str, self.keep_aspect_ratio, self.pad_value, self.pad_type,
                list(self.mean.flatten()), list(self.scale.flatten()), self.pixel_format, self.channel_format)
        logger.info(info_str)

    def load_config(self, input_op):
        shape = Operation.shape(input_op)
        self.net_input_dims = []
        if len(shape) >= 3:
            self.net_input_dims = shape[-2:]
            self.channel_num = shape[-3]
            self.batch_size = shape[0]
        else:
            print('error, len(input_op.shape) < 3, maybe have some error')
            exit(1)

        self.input_name = Operation.name(input_op)
        if 'preprocess' not in input_op.attributes:
            self.resize_dims = self.net_input_dims
            return

        attrs =Operation.dictattr(input_op, 'preprocess')
        self.pixel_format = Operation.str(attrs['pixel_format'])
        self.channel_format = Operation.str(attrs['channel_format'])
        if self.channel_format == 'nhwc':
            self.net_input_dims = shape[1:-1]
            self.channel_num = shape[-1]
        self.keep_aspect_ratio = Operation.bool(attrs['keep_aspect_ratio'])
        self.pad_value = Operation.int(attrs['pad_value'])
        self.pad_type = Operation.str(attrs['pad_type'])
        self.resize_dims = Operation.int_array(attrs['resize_dims'])
        if len(self.resize_dims) == 0:
            self.resize_dims = self.net_input_dims
        self.mean = np.array(Operation.fp_array(attrs['mean'])).astype(np.float32)
        self.mean = self.mean[np.newaxis, :,np.newaxis, np.newaxis]
        self.scale = np.array(Operation.fp_array(attrs['scale'])).astype(np.float32)
        self.scale = self.scale[np.newaxis, :,np.newaxis, np.newaxis]
        self.crop_method = 'center'
        if self.channel_format == 'nhwc':
            self.mean = self.mean.transpose(0, 2, 3, 1)
            self.scale = self.scale.transpose(0, 2, 3, 1)

        format_str = "\n  load_config Preprocess args : \n" + \
               "\tresize_dims           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\tpad_value             : {}\n" + \
               "\tpad_type              : {}\n" + \
               "\tinput_dims            : {}\n" + \
               "\t--------------------------\n" + \
               "\tmean                  : {}\n" + \
               "\tscale                 : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n" + \
               "\tchannel_format        : {}\n"
        logger.info(format_str.format(self.resize_dims, self.keep_aspect_ratio, self.pad_value, self.pad_type,
                self.net_input_dims,
                list(self.mean.flatten()), list(self.scale.flatten()),
                self.pixel_format,
                self.channel_format))

    def to_dict(self):
        return {
            'resize_dims': self.resize_dims,
            'keep_aspect_ratio': self.keep_aspect_ratio,
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
        start_h = (h // 2) -(crop_h // 2)
        start_w = (w // 2) - (crop_w // 2)
        img = img[:, :, start_h : (start_h + crop_h),
                     start_w : (start_w + crop_w)]
        return img

    def __load_image_and_resize(self, input):
        image = None
        image_path = str(input).rstrip()
        if not os.path.exists(image_path):
            print("{} doesn't existed !!!".format(image_path))
            exit(1)

        if self.channel_num == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        elif self.channel_num == 3:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        elif self.channel_num == 4:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image.shape[-1] != 4:
                image = PIL.Image.open(image_path).convert('RGBA')
                image = np.array(image)
        ratio = min(self.net_input_dims[0] / image.shape[0], self.net_input_dims[1] / image.shape[1])
        if self.keep_aspect_ratio:
            image = ImageResizeTool.letterbox_resize(
                image, self.resize_dims[0], self.resize_dims[1], self.pad_value, self.pad_type)
        else:
            image = ImageResizeTool.stretch_resize(
                image, self.resize_dims[0], self.resize_dims[1])

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

    def run(self, input):
        # load and resize image, the output image is chw format.
        x_list = []
        self.ratio_list = []
        for path in input.split(','):
            x,ratio = self.__load_image_and_resize(path)
            x = np.expand_dims(x, axis=0)
            x_list.append(x)
            self.ratio_list.append(ratio)
        x = np.concatenate(x_list, axis = 0)
        # take center crop if needed
        if self.resize_dims != self.net_input_dims:
            if self.crop_method == "right":
                x = self.__right_crop(x, self.net_input_dims)
            else:
                x = self.__center_crop(x, self.net_input_dims)

        x = x.astype(np.float32)
        if self.pixel_format == 'gray':
            self.mean = self.mean[:,:1,:,:]
            self.scale = self.scale[:,:1,:,:]
        elif self.pixel_format == 'rgb':
            x = x[:, [2, 1, 0], :, :]
        x = (x  - self.mean)* self.scale

        if self.channel_format == 'nhwc':
            x = np.transpose(x, (0, 2, 3, 1))

        if len(input.split(',')) == 1:
            x = np.repeat(x, self.batch_size, axis=0)
        return x
