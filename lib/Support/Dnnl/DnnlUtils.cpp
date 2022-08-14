
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
using namespace dnnl;
namespace tpu_mlir {

void post_relu(primitive_attr &attr, bool &do_relu, double &relu_limit)
{
  post_ops ops;
  if (do_relu) {
    const float ops_scale = 1.f;
    float ops_alpha = 0.f;
    const float ops_beta = 0.f;
    if (relu_limit > 0.f) {
      ops_alpha = relu_limit;
      ops.append_eltwise(ops_scale, algorithm::eltwise_bounded_relu, ops_alpha, ops_beta);
    } else {
      ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    }
    attr.set_post_ops(ops);
  }
}
} // namespace tpu_mlir
