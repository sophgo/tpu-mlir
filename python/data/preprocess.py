import os
import PIL
import numpy as np
import cv2
import argparse
from enum import Enum
from utils.log_setting import setup_logger
from utils.mlir_parser import *

logger = setup_logger('root', log_level="INFO")

pixel_format_attributes = {
    'rgb_planar':    (3, 'rgb'),
    'rgb_packed':    (3, 'rgb'),
    'bgr_planar':    (3, 'bgr'),
    'bgr_packed':    (3, 'bgr'),
    'gray':          (1,  ''),
    'rgba_planar':   (4, 'rgba')
}

# fix bool bug of argparse
def str2bool(v):
  return v.lower() in ("yes", "true", "1")

class ImageResizeTool:
    @staticmethod
    def stretch_resize(image, h, w):
        return cv2.resize(image, (w, h)) # w,h

    @staticmethod
    def letterbox_resize(image, h, w):
        ih = image.shape[0]
        iw = image.shape[1]
        scale = min(float(w) / iw, float(h) / ih)
        rescale_w = int(iw * scale)
        rescale_h = int(ih * scale)
        resized_img = cv2.resize(image, (rescale_w, rescale_h))
        paste_w = (w - rescale_w) // 2
        paste_h = (h - rescale_h) // 2
        if image.ndim == 3 and image.shape[2] == 3:
            new_image = np.full((h, w, 3), 0, dtype=image.dtype)
            new_image[paste_h:paste_h + rescale_h,
                      paste_w: paste_w + rescale_w, :] = resized_img
            return new_image
        elif image.ndim == 2:
            new_image = np.full((h, w), 0, dtype=image.dtype)
            new_image[paste_h:paste_h + rescale_h,
                      paste_w: paste_w + rescale_w] = resized_img
            return new_image
        raise RuntimeError("invalid image shape:{}".format(image.shape))

def add_preprocess_parser(parser):
    parser.add_argument("--resize_dims", type=str,
                        help="Image was resize to fixed 'h,w', default is same as net input dims")
    parser.add_argument("--keep_aspect_ratio", type=str2bool, default=False,
                        help="Resize image by keeping same ratio, any areas which" +
                             "are not taken are filled with 0")
    parser.add_argument("--mean", default='0,0,0', help="Per Channel image mean values")
    parser.add_argument("--scale", default='1,1,1', help="Per Channel image scale values")
    parser.add_argument("--pixel_format", choices=list(pixel_format_attributes.keys()), default='bgr_planar',
                        help='fixel format of output data that sent into model')
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
               mean='0,0,0', scale='1,1,1', pixel_format='bgr_planar', **ignored):
        if net_input_dims:
            self.net_input_dims = [int(s) for s in net_input_dims.split(',')]
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
        self.pixel_format = pixel_format
        if self.pixel_format not in pixel_format_attributes:
            raise RuntimeError("{} unsupported pixel format".format(pixel_format))

        self.input_name = 'input'
        self.channel_num = pixel_format_attributes[self.pixel_format][0]
        self.channel_order = pixel_format_attributes[self.pixel_format][1]

        _mean = np.array([float(s) for s in mean.split(',')], dtype=np.float32)
        assert(_mean.size >= self.channel_num)
        _mean = _mean[:self.channel_num]
        self.perchannel_mean = _mean[np.newaxis, :, np.newaxis, np.newaxis]
        _scale = np.array([float(s) for s in scale.split(',')], dtype=np.float32)
        assert(_scale.size >= self.channel_num)
        _scale = _scale[:self.channel_num]
        self.perchannel_scale = _scale[np.newaxis, :, np.newaxis, np.newaxis]

        info_str = \
            "\n\t _____________________________________________________________________ \n" + \
            "\t| preprocess:                                                           |\n" + \
            "\t|   (x - mean) * scale                                                  |\n" + \
            "\t'-----------------------------------------------------------------------'\n"

        format_str = "  config Preprocess args : \n" + \
               "\tresize_dims           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\t--------------------------\n" + \
               "\tperchannel_scale      : {}\n" + \
               "\tperchannel_mean       : {}\n" + \
               "\t   mean               : {}\n" + \
               "\t   scale              : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n"
        resize_dims_str = resize_dims if resize_dims is not None else 'same to net input dims'
        info_str += format_str.format(resize_dims_str, self.keep_aspect_ratio,
                list(self.perchannel_scale.flatten()), list(self.perchannel_mean.flatten()),
                list(_mean.flatten()), list(_scale.flatten()), self.pixel_format)
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
        self.channel_order = pixel_format_attributes[self.pixel_format][1]
        assert pixel_format_attributes[self.pixel_format][0] == self.channel_num
        self.keep_aspect_ratio = Operation.bool(attrs['keep_aspect_ratio'])
        self.resize_dims = Operation.int_array(attrs['resize_dims'])
        if len(self.resize_dims) == 0:
            self.resize_dims = self.net_input_dims
        self.perchannel_mean = np.array(Operation.fp_array(attrs['mean'])).astype(np.float32)
        self.perchannel_mean = self.perchannel_mean[np.newaxis, :,np.newaxis, np.newaxis]
        self.perchannel_scale = np.array(Operation.fp_array(attrs['scale'])).astype(np.float32)
        self.perchannel_scale = self.perchannel_scale[np.newaxis, :,np.newaxis, np.newaxis]
        self.crop_method = 'center'

        format_str = "\n  load_config Preprocess args : \n" + \
               "\tresize_dims           : {}\n" + \
               "\tkeep_aspect_ratio     : {}\n" + \
               "\t--------------------------\n" + \
               "\tperchannel_scale      : {}\n" + \
               "\tperchannel_mean       : {}\n" + \
               "\t--------------------------\n" + \
               "\tpixel_format          : {}\n"
        logger.info(format_str.format(self.resize_dims, self.keep_aspect_ratio,
                list(self.perchannel_scale.flatten()), list(self.perchannel_mean.flatten()),
                self.pixel_format))

    def to_dict(self):
        return {
            'resize_dims': self.resize_dims,
            'keep_aspect_ratio': self.keep_aspect_ratio,
            'perchannel_mean': list(self.perchannel_mean.flatten()),
            'perchannel_scale': list(self.perchannel_scale.flatten()),
            'pixel_format': self.pixel_format
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

        if self.keep_aspect_ratio:
            image = ImageResizeTool.letterbox_resize(
                image, self.resize_dims[0], self.resize_dims[1])
        else:
            image = ImageResizeTool.stretch_resize(
                image, self.resize_dims[0], self.resize_dims[1])

        if self.channel_num == 1:
            # if grapscale image, expand dim to (1, h, w)
            image = np.expand_dims(image, axis=0)
        else:
            # opencv read image data format is hwc, tranpose to chw
            image = np.transpose(image, (2, 0, 1))
        return image

    def run(self, input):
        # load and resize image, the output image is chw format.
        x_list = []
        for path in input.split(','):
            x = self.__load_image_and_resize(path)
            x = np.expand_dims(x, axis=0)
            x_list.append(x)
        x = np.concatenate(x_list, axis = 0)

        # take center crop if needed
        if self.resize_dims != self.net_input_dims:
            if self.crop_method == "right":
                x = self.__right_crop(x, self.net_input_dims)
            else:
                x = self.__center_crop(x, self.net_input_dims)

        x = x.astype(np.float32)
        if self.pixel_format == 'gray':
            self.perchannel_mean = self.perchannel_mean[:,:1,:,:]
            self.perchannel_scale = self.perchannel_scale[:,:1,:,:]
            x = (x  - self.perchannel_mean)* self.perchannel_scale
        elif self.pixel_format == 'rgba_planar' or self.pixel_format == 'rgb_planar' \
            or self.pixel_format == 'bgr_planar':
            if self.channel_order == 'rgb':
                x = x[:, [2, 1, 0], :, :]
            x = (x  - self.perchannel_mean)* self.perchannel_scale
        elif self.pixel_format == 'rgb_packed' or self.pixel_format == 'bgr_packed':
            if self.channel_order == "rgb":
                x = x[:, [2, 1, 0], :, :]
            x = (x  - self.perchannel_mean)* self.perchannel_scale
            x = np.transpose(x, (0, 2, 3, 1))
        else:
            logger.info("unsupported pixel format");
            assert(0)

        print('wxc2:', self.batch_size)
        if len(input.split(',')) == 1:
            x = np.repeat(x, self.batch_size, axis=0)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()

    preprocesser = preprocess()

    preprocesser.config(net_input_dims='244,224', pixel_format='bgr_planar')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (224, 244)) # w,h
    y=np.transpose(y, (2, 0, 1))
    if np.any(x != y):
        raise Exception("1. BGR PLANAR test failed")
    logger.info("1. BGR PLANAR test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='rgb_planar')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    y=cv2.resize(y, (224, 244)) # w,h
    y=np.transpose(y, (2, 0, 1))
    if np.any(x != y):
        raise Exception("2. RGB PLANAR test failed")
    logger.info("2. RGB PLANAR test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='bgr_packed')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("3. BGR PACKED test failed")
    logger.info("3. BGR PACKED test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='rgb_packed')
    x = preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("RGB PACKED test failed")
    logger.info("4. RGB PACKED test passed!!")

    preprocesser.config(net_input_dims='244,224', pixel_format='gray')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    y=cv2.resize(y, (224, 244)) # w,h
    if np.any(x != y):
        raise Exception("5. gray test failed")
    logger.info("5. gray test passed!!")

    preprocesser.config(net_input_dims='244,224', resize_dims='443,424',
                        crop_method='center', pixel_format='bgr_packed')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image)
    y=cv2.resize(y, (424, 443))
    h_offset = (443 - 244) // 2
    w_offset = (424 - 224) // 2
    y = y[h_offset:h_offset + 244, w_offset : w_offset + 224]
    if np.any(x != y):
        raise Exception("6. Center crop test failed")
    logger.info("6. Center Crop test passed!!")

    preprocesser.config(net_input_dims='244,224', keep_aspect_ratio=True,
                        pixel_format='bgr_packed')
    x=preprocesser.run(args.image)
    y=cv2.imread(args.image)
    ih, iw, _ = y.shape
    w, h = (224, 244)
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    y0 = cv2.resize(y, (nw,nh))
    y = np.full((h, w, 3), 0, dtype='uint8')
    y[(h - nh) // 2:(h - nh) // 2 + nh,
            (w - nw) // 2:(w - nw) // 2 + nw, :] = y0
    if np.any(x != y):
        raise Exception("6. keep ratio resize test failed")
    logger.info("6. keep ratio resize test passed!!")
