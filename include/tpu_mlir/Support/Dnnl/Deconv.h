//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Support/LLVM.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "llvm/ADT/SmallVector.h"
#include "tpu_mlir/Support/AttrStruct.h"
using namespace dnnl;
namespace tpu_mlir {

class Deconv {
public:
  Deconv();
  ~Deconv();
  void setup(float *input, float *weight, float *bias, float *output,
             const deconv_attr_t &attr, int izp = 0);

  void run();

private:
  void pad_init(float *input, deconv_attr_t &attr, int izp);

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

std::optional<llvm::SmallVector<float, 4>>
DeconvSlice(int64_t out_idx, int64_t out_slice, int64_t stride, int64_t filter,
            int64_t ih, int64_t pad);
} // namespace tpu_mlir
