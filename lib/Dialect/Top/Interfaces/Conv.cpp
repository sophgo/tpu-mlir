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

void top::ConvOp::parseParam(void *param) {
  conv_attr_t *p = (conv_attr_t *)param;
  memset(p, 0, sizeof(conv_attr_t));
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  p->do_relu = this->do_relu();
  p->relu_limit = relu_limit().convertToDouble();
  p->has_bias = !bias().getType().isa<NoneType>();
  auto kernel = Module::getI64Array(kernel_shape());
  auto pads_v = Module::getI64Array(pads());
  auto strides_v = Module::getI64Array(strides());
  auto dilation = Module::getI64Array(dilations(), kernel->size(), 1);
  auto ins = Module::getI64Array(inserts(), kernel->size(), 0);
  p->n = i_s[0];
  p->ic = i_s[1];
  p->oc = o_s[1];
  if (kernel->size() == 3) {
    p->id = i_s[2];
    p->ih = i_s[3];
    p->iw = i_s[4];
    p->od = o_s[2];
    p->oh = o_s[3];
    p->ow = o_s[4];
    p->kd = kernel->at(0);
    p->kh = kernel->at(1);
    p->kw = kernel->at(2);
    p->pdf = pads_v->at(0);
    p->pht = pads_v->at(1);
    p->pwl = pads_v->at(2);
    p->pdb = pads_v->at(3);
    p->phb = pads_v->at(4);
    p->pwr = pads_v->at(5);
    p->sd = strides_v->at(0);
    p->sh = strides_v->at(1);
    p->sw = strides_v->at(2);
    p->dd = dilation->at(0);
    p->dh = dilation->at(1);
    p->dw = dilation->at(2);
    p->ins_d = ins->at(0);
    p->ins_h = ins->at(1);
    p->ins_w = ins->at(2);
  } else if (kernel->size() == 2) {
    p->id = p->od = p->kd = p->dd = p->sd = 1;
    p->ih = i_s[2];
    p->iw = i_s[3];
    p->oh = o_s[2];
    p->ow = o_s[3];
    p->kh = kernel->at(0);
    p->kw = kernel->at(1);
    p->pht = pads_v->at(0);
    p->pwl = pads_v->at(1);
    p->phb = pads_v->at(2);
    p->pwr = pads_v->at(3);
    p->sh = strides_v->at(0);
    p->sw = strides_v->at(1);
    p->dh = dilation->at(0);
    p->dw = dilation->at(1);
    p->ins_h = ins->at(0);
    p->ins_w = ins->at(1);
  } else if (kernel->size() == 1) {
    p->id = p->od = p->kd = p->dd = p->sd = 1;
    p->iw = p->ow = p->kw = p->dw = p->sw = 1;
    p->ih = i_s[2];
    p->oh = o_s[2];
    p->kh = kernel->at(0);
    p->pht = pads_v->at(0);
    p->phb = pads_v->at(1);
    p->sh = strides_v->at(0);
    p->dh = dilation->at(0);
    p->ins_h = ins->at(0);
  }
  assert(p->ins_d == 0 && p->ins_h == 0 && p->ins_w == 0);
  p->groups = group();
  p->is_dw = (p->oc == p->ic && p->oc == p->groups && p->groups > 1);
}

int64_t top::ConvOp::getFLOPs() {
  conv_attr_t attr = {0};
  parseParam(&attr);
  auto extra = attr.has_bias ? 1 : 0 + attr.do_relu ? 1 : 0;
  return Module::getNumElements(output()) *
         (attr.kd * attr.kh * attr.kw * attr.ic / attr.groups * 2 + extra);
}

LogicalResult top::ConvOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  conv_attr_t attr = {0};
  parseParam(&attr);
  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  p.handle = (void *)conv;
  return success();
}

void top::ConvOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult top::ConvOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  return success();
}
