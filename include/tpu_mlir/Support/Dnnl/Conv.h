//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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
  int64_t kd, dd, sd, ins_d;
  int64_t kh, dh, sh, ins_h;
  int64_t kw, dw, sw, ins_w;
  int64_t pdf, pdb;
  int64_t pht, phb;
  int64_t pwl, pwr;
  int64_t groups;
  int64_t pad_value;
  bool has_bias;
  bool is_dw;
  bool do_relu;
  double relu_limit;
} conv_attr_t;

class Conv {
public:
  Conv();
  ~Conv();

  void pad_init(float *input, conv_attr_t &attr);
  void setup(float *input, float *weight, float *bias, float *output,
             conv_attr_t attr);

  void run();

private:
  engine eng;
  stream eng_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  convolution_forward::primitive_desc conv_prim_desc;
  memory prim_filter_memory;
  memory prim_bias_memory;
  memory::dims src_shape;
  memory::dims dst_shape;
  float *p_input;
  float *origin_input;
  std::shared_ptr<std::vector<float>> input_after_pad;
  conv_attr_t _attr;
};
} // namespace tpu_mlir
