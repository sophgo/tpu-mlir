
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
using namespace dnnl;
namespace tpu_mlir {

void post_relu(primitive_attr &attr, bool &do_relu, double &relu_limit) {
  post_ops ops;
  if (do_relu) {
    if (relu_limit > 0.f) {
      ops.append_eltwise(algorithm::eltwise_clip, 0.0f, relu_limit);
    } else {
      ops.append_eltwise(algorithm::eltwise_relu, 0.0f, 0.0f);
    }
    attr.set_post_ops(ops);
  }
}
} // namespace tpu_mlir
