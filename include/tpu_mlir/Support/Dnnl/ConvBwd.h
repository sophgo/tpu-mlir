//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

namespace tpu_mlir {

class ConvBwd {
public:
    ConvBwd();
    ~ConvBwd();

    void setup(float *src_data, float *weights_data, float *dst_data,
               float *grad_input, float *grad_weight, float *grad_bias,
               int batch_size, int in_channels, int out_channels,
               int height, int width, int kernel_h, int kernel_w,
               int out_h, int out_w, int stride_h, int stride_w,
               int pad_h, int pad_w,
               bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias);

    void run();

private:
    engine eng;
    stream eng_stream;

    memory::dims src_shape;
    memory::dims weights_shape;
    memory::dims dst_shape;
    memory::dims strides;
    memory::dims padding_l;
    memory::dims padding_r;

    memory src_mem;
    memory weights_mem;
    memory dst_mem;
    memory grad_input_mem;
    memory grad_weight_mem;
    memory grad_bias_mem;

    convolution_forward::primitive_desc conv_fwd_pd;
    convolution_backward_data::primitive_desc conv_bwd_data_pd;
    convolution_backward_weights::primitive_desc conv_bwd_weights_pd;

    primitive conv_bwd_data_prim;
    primitive conv_bwd_weights_prim;

    std::vector<primitive> net_bw;
    std::vector<std::unordered_map<int, memory>> net_bw_args;

    bool compute_grad_input;
    bool compute_grad_weight;
    bool compute_grad_bias;
};

} // namespace tpu_mlir