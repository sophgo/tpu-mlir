#include "tpu_mlir/Support/Dnnl/ConvBwd.h"
#include <iostream>

namespace tpu_mlir {

ConvBwd::ConvBwd() : eng(engine::kind::cpu, 0), eng_stream(eng) {
  // Initialize engine and stream
}

ConvBwd::~ConvBwd() {
  // Cleanup resources if necessary
}

void ConvBwd::setup(float *src_data, float *weights_data, float *dst_data,
                    float *grad_input, float *grad_weight, float *grad_bias,
                    int batch_size, int in_channels, int out_channels,
                    int height, int width, int kernel_h, int kernel_w,
                    int out_h, int out_w, int stride_h, int stride_w,
                    int pad_h_t, int pad_w_l, int pad_h_b, int pad_w_r,
                    bool compute_grad_input, bool compute_grad_weight,
                    bool compute_grad_bias, int dilation_h, int dilation_w) {
  // Set up shapes and parameters
  src_shape = {batch_size, in_channels, height, width};
  weights_shape = {out_channels, in_channels, kernel_h, kernel_w};
  dst_shape = {batch_size, out_channels, out_h, out_w};
  strides = {stride_h, stride_w};
  padding_l = {pad_h_t, pad_w_l};
  padding_r = {pad_h_b, pad_w_r};
  dilation = {dilation_h - 1, dilation_w - 1};

  // Create memory descriptors
  auto src_md =
      memory::desc(src_shape, memory::data_type::f32, memory::format_tag::nchw);
  auto weights_md = memory::desc(weights_shape, memory::data_type::f32,
                                 memory::format_tag::oihw);
  auto dst_md =
      memory::desc(dst_shape, memory::data_type::f32, memory::format_tag::nchw);

  // Create forward convolution primitive descriptor
  conv_fwd_pd = convolution_forward::primitive_desc(
      eng, prop_kind::forward_training, algorithm::convolution_auto, src_md,
      weights_md, dst_md, strides, dilation, padding_l, padding_r);

  // Create backward data convolution primitive descriptor
  conv_bwd_data_pd = convolution_backward_data::primitive_desc(
      eng, algorithm::convolution_direct, src_md, weights_md, dst_md, strides,
      dilation, padding_l, padding_r, conv_fwd_pd);

  // Create backward weights convolution primitive descriptor
  conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(
      eng, algorithm::convolution_direct, src_md, weights_md, dst_md, strides,
      dilation, padding_l, padding_r, conv_fwd_pd);

  // Create memory objects
  src_mem = memory(src_md, eng, src_data);
  weights_mem = memory(weights_md, eng, weights_data);
  dst_mem = memory(dst_md, eng, dst_data);

  // Create gradient memory objects
  grad_input_mem = memory(src_md, eng, grad_input);
  grad_weight_mem = memory(weights_md, eng, grad_weight);
  grad_bias_mem =
      memory({{out_channels}, memory::data_type::f32, memory::format_tag::x},
             eng, grad_bias);

  // Create backward data primitive
  conv_bwd_data_prim = convolution_backward_data(conv_bwd_data_pd);

  // Create backward weights primitive
  conv_bwd_weights_prim = convolution_backward_weights(conv_bwd_weights_pd);

  // Set up backward network
  net_bw.clear();
  net_bw_args.clear();

  if (compute_grad_input) {
    net_bw.push_back(conv_bwd_data_prim);
    net_bw_args.push_back({{DNNL_ARG_DIFF_DST, dst_mem},
                           {DNNL_ARG_WEIGHTS, weights_mem},
                           {DNNL_ARG_DIFF_SRC, grad_input_mem}});
  }

  if (compute_grad_weight || compute_grad_bias) {
    net_bw.push_back(conv_bwd_weights_prim);
    net_bw_args.push_back({{DNNL_ARG_DIFF_DST, dst_mem},
                           {DNNL_ARG_SRC, src_mem},
                           {DNNL_ARG_DIFF_WEIGHTS, grad_weight_mem},
                           {DNNL_ARG_DIFF_BIAS, grad_bias_mem}});
  }

  this->compute_grad_input = compute_grad_input;
  this->compute_grad_weight = compute_grad_weight;
  this->compute_grad_bias = compute_grad_bias;
}

void ConvBwd::run() {
  // Execute backward network
  for (size_t i = 0; i < net_bw.size(); ++i) {
    net_bw[i].execute(eng_stream, net_bw_args[i]);
  }
  eng_stream.wait();
}

} // namespace tpu_mlir