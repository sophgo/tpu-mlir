//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void tpu::MaxPool3DOp::parseParam(void *param) {
  pool_attr_t *p = (pool_attr_t *)param;
  memset(p, 0, sizeof(pool_attr_t));
  assert(kernel_shape().size() == 3);
  auto ishape = input().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = output().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = Module::getI64Array(kernel_shape());
  auto stride = Module::getI64Array(strides());
  auto pad = Module::getI64Array(pads());

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
  p->pad_value = pad_value();
  p->do_relu = do_relu();
  p->is_global = p->id == p->kd && p->ih == p->kh && p->iw == p->kw &&
                 p->od == 1 && p->oh == 1 && p->ow == 1;
  p->count_include_pad = count_include_pad();
}

LogicalResult tpu::MaxPool3DOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  pool_attr_t attrs;
  parseParam(&attrs);

  int izp = 0;
  auto dtype = input().getType().cast<RankedTensorType>().getElementType();
  if (dtype.isa<quant::UniformQuantizedType>()) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }

  pooling->setup(p.inputs[0], p.outputs[0], attrs, false, izp);
  p.handle = (void *)pooling;
  return success();
}

void tpu::MaxPool3DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::MaxPool3DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (do_relu()) {
    function_relu(p.outputs[0], p.outputs[0], Module::getNumElements(output()),
                  0.f, Module::getStorageType(output()));
  }
  return success();
}

LogicalResult tpu::MaxPool3DOp::LocalGenSupport() {
  auto stride = Module::getI64Array(strides());
  if (stride->at(0) > 15 || stride->at(1) > 15 || stride->at(2) > 15) {
    return failure();
  }
  /// no local layer now
  return failure();
}

LogicalResult tpu::MaxPool3DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  pool_attr_t attrs;
  parseParam(&attrs);
  in_slice = (out_slice - 1) * attrs.sh + attrs.kh;
  in_idx = out_idx * attrs.sh - attrs.pad_h;
  LocalGenInterface::fixSlice(in_idx, in_slice, attrs.ih);
  return success();
}
