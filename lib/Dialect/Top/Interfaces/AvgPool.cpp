//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::AvgPoolOp::getFLOPs() {
  pool_attr_t attr;
  parseParam(&attr);
  return Module::getNumElements(output()) *
         (attr.kd * attr.kh * attr.kw + attr.do_relu ? 1 : 0);
}

void top::AvgPoolOp::parseParam(void *param) {
  pool_attr_t *p = (pool_attr_t *)param;
  memset(p, 0, sizeof(pool_attr_t));
  auto ishape = input().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = output().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = Module::getI64Array(kernel_shape());
  auto stride = Module::getI64Array(strides());
  auto pad = Module::getI64Array(pads());
  if (kernel_shape().size() == 3) {
    p->n = ishape[0];
    p->c = ishape[1];
    p->id = ishape[2];
    p->ih = ishape[3];
    p->iw = ishape[4];
    p->od = oshape[2];
    p->oh = oshape[3];
    p->ow = oshape[4];
    p->kd = kernel->at(0);
    p->kh = kernel->at(1);
    p->kw = kernel->at(2);
    p->sd = stride->at(0);
    p->sh = stride->at(1);
    p->sw = stride->at(2);
    p->pad_d = pad->at(0);
    p->pad_h = pad->at(1);
    p->pad_w = pad->at(2);
    p->pad_d_after = pad->at(3);
    p->pad_h_after = pad->at(4);
    p->pad_w_after = pad->at(5);
  } else if (kernel_shape().size() == 2) {
    p->id = 1;
    p->od = 1;
    p->kd = 1;
    p->sd = 1;
    Module::getNCHW(ishape, p->n, p->c, p->ih, p->iw);
    Module::getNCHW(oshape, p->n, p->c, p->oh, p->ow);
    p->kh = kernel->at(0);
    p->kw = kernel->at(1);
    p->sh = stride->at(0);
    p->sw = stride->at(1);
    p->pad_h = pad->at(0);
    p->pad_w = pad->at(1);
    p->pad_h_after = pad->at(2);
    p->pad_w_after = pad->at(3);
  } else if (kernel_shape().size() == 1) {
    p->id = 1;
    p->od = 1;
    p->kd = 1;
    p->kw = 1;
    p->sd = 1;
    p->sw = 1;
    Module::getNCHW(ishape, p->n, p->c, p->ih, p->iw);
    Module::getNCHW(oshape, p->n, p->c, p->oh, p->ow);
    p->kh = kernel->at(0);
    p->sh = stride->at(0);
    p->pad_h = pad->at(0);
    p->pad_h_after = pad->at(1);
  }
  p->pad_value = pad_value();
  p->do_relu = do_relu();
  p->is_global = p->id == p->kd && p->ih == p->kh && p->iw == p->kw &&
                 p->od == 1 && p->oh == 1 && p->ow == 1;
  p->count_include_pad = count_include_pad();
}

LogicalResult top::AvgPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  pool_attr_t attr;
  parseParam(&attr);
  pooling->setup(p.inputs[0], p.outputs[0], attr, true);
  p.handle = (void *)pooling;
  return success();
}

void top::AvgPoolOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::AvgPoolOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (do_relu()) {
    auto limit = relu_limit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], Module::getNumElements(output()),
                  limit);
  }
  return success();
}
