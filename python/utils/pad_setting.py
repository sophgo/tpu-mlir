from math import ceil


def set_auto_pad(auto_pad, input_shape, kernel_shape, strides):
    if isinstance(auto_pad, str):
        pad_method = auto_pad
    else:
        pad_method = auto_pad.decode('utf-8')
    if pad_method == "SAME_UPPER":
        padding_along_h = get_TF_SAME_Padding(input_shape[2], kernel_shape[0], strides[0])
        padding_along_w = get_TF_SAME_Padding(input_shape[3], kernel_shape[1], strides[1])
        padding_t = padding_along_h // 2
        padding_l = padding_along_w // 2
        padding_b = padding_along_h - padding_t
        padding_r = padding_along_w - padding_l
        pads = [padding_t, padding_l, padding_b, padding_r]
    elif pad_method == "SAME_LOWER":
        # the extra padding is added at the beginning for SAME_LOWER.
        padding_along_h = get_TF_SAME_Padding(input_shape[2], kernel_shape[0], strides[0])
        padding_along_w = get_TF_SAME_Padding(input_shape[3], kernel_shape[1], strides[1])
        padding_b = padding_along_h // 2
        padding_r = padding_along_w // 2
        padding_t = padding_along_h - padding_b
        padding_l = padding_along_w - padding_r
        pads = [padding_t, padding_l, padding_b, padding_r]
    elif pad_method == "NOTSET":
        pads = []
    elif pad_method == "VALID":
        pads = []
    else:
        raise RuntimeError("Not support {} pad method".format(pad_method))
    return pads

def set_caffe_pad(input_shape, output_shape, kernel_shape, strides, leading_pad):
    padding_along_h = (output_shape[2] - 1) * strides[0] + kernel_shape[0] - input_shape[2]
    padding_along_w = (output_shape[3] - 1) * strides[1] + kernel_shape[1] - input_shape[3]
    pad_t, pad_l = leading_pad
    pad_b = padding_along_h - pad_t
    if pad_b < 0:
        pad_b = 0
    pad_r = padding_along_w - pad_l
    if pad_r < 0:
        pad_r = 0
    return [pad_t, pad_l, pad_b, pad_r]

def get_TF_SAME_Padding(input_spatial_shape, kernel, stride):
    """
    If padding == "SAME":
      output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])
    """
    output_spatial_shape = int(ceil(float(input_spatial_shape) / float(stride)))
    if input_spatial_shape % stride == 0:
        pad_along = max((kernel - stride), 0)
    else:
        pad_along = max(kernel - (input_spatial_shape % stride), 0)

    return pad_along
