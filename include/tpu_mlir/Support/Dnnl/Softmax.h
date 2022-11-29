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
  uint64_t axis;
  bool log;
  memory::dims src_shape;
  memory::dims dst_shape;
} softmax_attr_t;

class Softmax{
public:
  Softmax();
  void setup(float *input, float *output, softmax_attr_t &attr);
  void run();
  ~Softmax() = default;
private:
  engine eng;
  stream eng_stream;
  primitive softmax_prim;
  std::unordered_map<int, memory> softmax_args;
  float *p_input;
  float *p_output;
  softmax_attr_t attr_;
};

}
