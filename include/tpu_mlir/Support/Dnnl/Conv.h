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
class Conv {
public:
  Conv();
  ~Conv();

  void pad_init(float *input, int n, int ic, int ih, int iw, int &pt, int &pb,
                int &pl, int &pr, int izp);
  void setup(float *input, float *weight, float *bias, float *output, int n,
             int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
             int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
             int g, bool do_relu, float relu_upper_limit, int izp = 0);

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
  int _n, _c, _h, _w;
  int _pt;
  int _pb;
  int _pl;
  int _pr;
  int _izp;
};
} // namespace tpu_mlir
