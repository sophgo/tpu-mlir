from math import ceil


def set_auto_pad(auto_pad, input_shape, kernel_shape, strides):
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
    else:
        raise RuntimeError("Not support conv {} pad method".format(pad_method))
    return pads


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
