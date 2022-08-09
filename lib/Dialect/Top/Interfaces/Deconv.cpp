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

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void top::DeconvOp::parseParam(void *param) {
  deconv_attr_t *p = (deconv_attr_t *)param;
  memset(p, 0, sizeof(deconv_attr_t));
  bool is_deconv3d = kernel_shape().size() == 3;
  auto ishape = input().getType().cast<RankedTensorType>().getShape();
  auto oshape = output().getType().cast<RankedTensorType>().getShape();
  auto kernel = Module::getI64Array(kernel_shape());
  auto stride = Module::getI64Array(strides());
  auto dilation = Module::getI64Array(dilations(), kernel_shape().size(), 1);
  auto pad = Module::getI64Array(pads());
  auto ins = Module::getI64Array(inserts(), kernel_shape().size(), 0);
  p->do_relu = do_relu();
  p->with_bias = !bias().getType().isa<NoneType>();
  p->g = group();
  if (is_deconv3d) {
    p->n = ishape[0];
    p->ic = ishape[1];
    p->id = ishape[2];
    p->ih = ishape[3];
    p->iw = ishape[4];
    p->oc = oshape[1];
    p->od = oshape[2];
    p->oh = oshape[3];
    p->ow = oshape[4];
    p->kd = kernel->at(0);
    p->kh = kernel->at(1);
    p->kw = kernel->at(2);
    p->sd = stride->at(0);
    p->sh = stride->at(1);
    p->sw = stride->at(2);
    p->dd = dilation->at(0);
    p->dh = dilation->at(1);
    p->dw = dilation->at(2);
    p->ins_d = ins->at(0);
    p->ins_h = ins->at(1);
    p->ins_w = ins->at(2);
    p->pad_d = pad->at(0);
    p->pad_h = pad->at(1);
    p->pad_w = pad->at(2);
    p->pad_d_after = pad->at(3);
    p->pad_h_after = pad->at(4);
    p->pad_w_after = pad->at(5);
  } else {
    p->n = ishape[0];
    p->ic = ishape[1];
    p->ih = ishape[2];
    p->iw = ishape[3];
    p->oc = oshape[1];
    p->oh = oshape[2];
    p->ow = oshape[3];
    p->kh = kernel->at(0);
    p->kw = kernel->at(1);
    p->sh = stride->at(0);
    p->sw = stride->at(1);
    p->dh = dilation->at(0);
    p->dw = dilation->at(1);
    p->ins_h = ins->at(0);
    p->ins_w = ins->at(1);
    p->pad_h = pad->at(0);
    p->pad_w = pad->at(1);
    p->pad_h_after = pad->at(2);
    p->pad_w_after = pad->at(3);
    p->id = 1;
    p->od = 1;
    p->kd = 1;
    p->sd = 1;
    p->dd = 1;
    p->ins_d = 0;
  }
  p->is_dw = (p->oc == p->ic && p->oc == p->g && p->g > 1);
  return;
}

int64_t top::DeconvOp::getFLOPs() {
  deconv_attr_t attr;
  parseParam(&attr);
  auto extra = attr.with_bias ? 1 : 0 + attr.do_relu ? 1 : 0;
  return Module::getNumElements(input()) *
         (attr.kw * attr.kw * attr.oc / attr.g * 2 + extra);
}

LogicalResult top::DeconvOp::init(InferenceParameter &p) {
  auto deconv = new Deconv();
  deconv_attr_t attr;
  parseParam(&attr);
  deconv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  p.handle = (void *)deconv;
  return success();
}

void top::DeconvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto deconv = (Deconv *)p.handle;
    delete deconv;
    p.handle = nullptr;
  }
}

LogicalResult top::DeconvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto deconv = (Deconv *)p.handle;
  deconv->run();
  return success();
}
