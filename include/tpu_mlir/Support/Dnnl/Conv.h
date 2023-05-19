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
#include "tpu_mlir/Support/AttrStruct.h"
using namespace dnnl;
namespace tpu_mlir {

class Conv {
public:
  Conv();
  ~Conv();
  void filter_init(float *weight, conv_attr_t &attr);
  void setup(float *input, float *weight, float *bias, float *output,
             conv_attr_t attr);
  void run();

  void diff_filter_init(memory::dims &filter_shape);
  void diff_bias_init(memory::dims &bias_shape);
  void diff_dst_init(memory::dims &dst_shape);
  void run_backw(void *dst_grd_input, void *weight_grd_output);
private:
  void activation_init(float *input, conv_attr_t &attr);
  void backward_weights_setup();

private:
  engine eng;
  stream eng_stream;
  convolution_forward::primitive_desc conv_prim_desc;
  primitive prim;
  std::shared_ptr<std::vector<float>> bias0;
  memory src_mem, filter_mem, bias_mem, dst_mem;
  memory::dims src_shape;
  memory::dims dst_shape;
  float *p_input, *p_weight;
  float *origin_input, *origin_weight;
  std::shared_ptr<std::vector<float>> input_after_pad, weight_after_zp;
  conv_attr_t _attr;

  bool backw_init;
  std::vector<primitive> net_bw;
  std::vector<std::unordered_map<int, memory>> net_bw_args;
  float *p_ginput, *p_goutput, *p_gweight, *p_gbias;
  std::shared_ptr<std::vector<float>> diff_filter, diff_bias, diff_dst;
};
} // namespace tpu_mlir
