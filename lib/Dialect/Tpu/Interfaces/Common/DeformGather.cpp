//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"

#include "tpu_mlir/Support/DeformConv2D.h"

using namespace tpu_mlir::backend;

deform_gather_attr_t tpu::DeformGatherOp::parseParam() {
  deform_gather_attr_t p = {0};
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  auto of_s = getOffset().getType().cast<RankedTensorType>().getShape();

  p.n = i_s[0];
  p.ic = i_s[1];
  p.ih = i_s.size() > 2 ? i_s[2] : 1;
  p.iw = i_s.size() > 3 ? i_s[3] : 1;
  p.oc = o_s[1];
  p.oh = o_s.size() > 2 ? o_s[2] : 1;
  p.ow = o_s.size() > 3 ? o_s[3] : 1;
  p.ofc = of_s[1];
  p.ofh = of_s.size() > 2 ? of_s[2] : 1;
  p.ofw = of_s.size() > 3 ? of_s[3] : 1;
  p.use_mask = getUseMask();
  if (p.use_mask) {
    auto mk_s = getMask().getType().cast<RankedTensorType>().getShape();
    p.mkc = mk_s[1];
    p.mkh = mk_s.size() > 2 ? mk_s[2] : 1;
    p.mkw = mk_s.size() > 3 ? mk_s[3] : 1;
  }

  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto pads_v = module::getI64Array(getPads());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  auto strides_v = module::getI64Array(getStrides());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dilation_v = module::getI64Array(getDilations(), 2, 1);
  p.dh = dilation_v->at(0);
  p.dw = dilation_v->at(1);
  p.deform_groups = getDeformGroup();

  return p;
}

LogicalResult tpu::DeformGatherOp::init(InferenceParameter &p) {
  auto num = module::getNumElements(getOutput()); // output size, buffer size
  float *buffer = new float[num];
  p.handle = (void *)buffer;
  return success();
}

void tpu::DeformGatherOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    float *buffer = (float *)p.handle;
    delete[] buffer;
    p.handle = nullptr;
  }
}

LogicalResult tpu::DeformGatherOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  processDeformGather(p, attr, p.outputs[0], false);
  return success();
}

bool tpu::DeformGatherOp::support_multi_core() { return false; }
