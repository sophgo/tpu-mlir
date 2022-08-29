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

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t id;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t od;
  int64_t oh;
  int64_t ow;
  int64_t kd;
  int64_t kh;
  int64_t kw;
  int64_t sd;
  int64_t sh;
  int64_t sw;
  int64_t dd;
  int64_t dh;
  int64_t dw;
  int64_t ins_d;
  int64_t ins_h;
  int64_t ins_w;
  int64_t pad_d;
  int64_t pad_d_after;
  int64_t pad_h;
  int64_t pad_h_after;
  int64_t pad_w;
  int64_t pad_w_after;
  int64_t output_pad_d;
  int64_t output_pad_h;
  int64_t output_pad_w;
  int64_t pad_value;
  int64_t g;
  bool with_bias;
  bool do_relu;
  double relu_limit;
  bool is_dw;
  bool pad_insert_is_const;
} deconv_attr_t;

class Deconv {
public:
  Deconv();
  ~Deconv();

  void pad_init(float *input, deconv_attr_t &attr, int izp);
  void setup(float *input, float *weight, float *bias, float *output,
             deconv_attr_t &attr, int izp = 0);

  void run();

public:
  int kd, kh, kw;

private:
  engine eng;
  stream eng_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  deconvolution_forward::primitive_desc deconv_prim_desc;
  convolution_forward::primitive_desc conv_prim_desc;
  memory prim_filter_memory;
  memory prim_bias_memory;
  memory::dims src_shape;
  memory::dims dst_shape;
  float *p_input;
  float *origin_input;
  std::shared_ptr<std::vector<float>> input_after_pad;
  std::shared_ptr<std::vector<float>> weight_rotated;
  deconv_attr_t _attrs;
  int _izp;
};
} // namespace tpu_mlir
