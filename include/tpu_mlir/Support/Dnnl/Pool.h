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

class Pooling {
public:
  Pooling();
  ~Pooling();
  void setup(float *input, float *output, pool_attr_t attr, bool is_avg,
             int izp = 0);
  void run();

private:
  void pad_init(float *input, pool_attr_t &attr, int izp);

public:
  int kd, kh, kw;

private:
  engine eng;
  stream eng_stream;
  primitive prim;
  memory src_mem, dst_mem;
  memory::dims src_shape;
  memory::dims dst_shape;
  float *p_input;
  float *origin_input;
  std::shared_ptr<std::vector<float>> input_after_pad;
  pool_attr_t _attrs;
  int _izp;
};
} // namespace tpu_mlir
