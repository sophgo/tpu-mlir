//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

void tpu::Conv3DOp::parseParam(void *param) {
  conv_attr_t *p = (conv_attr_t *)param;
  memset(p, 0, sizeof(conv_attr_t));
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  p->do_relu = this->do_relu();
  p->relu_limit = relu_limit().convertToDouble();
  p->has_bias = with_bias();
  p->n = i_s[0];
  p->ic = i_s[1];
  p->id = i_s[2];
  p->ih = i_s[3];
  p->iw = i_s[4];
  p->oc = o_s[1];
  p->od = o_s[2];
  p->oh = o_s[3];
  p->ow = o_s[4];
  auto kernel = Module::getI64Array(kernel_shape());
  p->kd = kernel->at(0);
  p->kh = kernel->at(1);
  p->kw = kernel->at(2);
  auto pads_v = Module::getI64Array(pads());
  p->pdf = pads_v->at(0);
  p->pht = pads_v->at(1);
  p->pwl = pads_v->at(2);
  p->pdb = pads_v->at(3);
  p->phb = pads_v->at(4);
  p->pwr = pads_v->at(5);
  if (Quant::isUniformQuantized(input())) {
    p->pad_value = Quant::getUniformQuantizedType(input()).getZeroPoint();
  }
  if (Quant::isUniformQuantized(filter())) {
    p->kernel_zp = Quant::getUniformQuantizedType(filter()).getZeroPoint();
  }
  auto strides_v = Module::getI64Array(strides());
  p->sd = strides_v->at(0);
  p->sh = strides_v->at(1);
  p->sw = strides_v->at(2);
  auto dilation = Module::getI64Array(dilations(), 3, 1);
  p->dd = dilation->at(0);
  p->dh = dilation->at(1);
  p->dw = dilation->at(2);
  auto ins = Module::getI64Array(inserts(), 3, 0);
  p->ins_d = ins->at(0);
  p->ins_h = ins->at(1);
  p->ins_w = ins->at(2);
  assert(p->ins_d == 0 && p->ins_h == 0 && p->ins_w == 0);
  p->groups = group();
  p->is_dw = (p->oc == p->ic && p->oc == p->groups && p->groups > 1);
}
LogicalResult tpu::Conv3DOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  conv_attr_t attr = {0};
  parseParam(&attr);

  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  p.handle = (void *)conv;
  return success();
}

void tpu::Conv3DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::Conv3DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  // requant
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  }

  return success();
}

LogicalResult tpu::Conv3DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  conv_attr_t attr = {0};
  parseParam(&attr);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::Conv3DOp::LocalGenSupport() { return failure(); }

mlir::Type tpu::Conv3DOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}
