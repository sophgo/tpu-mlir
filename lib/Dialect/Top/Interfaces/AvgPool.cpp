//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"


bool top::AvgPoolOp::isEltwise() {
  return false;
}

int64_t top::AvgPoolOp::getFLOPs() {
  auto attr = parseParam();
  return module::getNumElements(getOutput()) *
         (attr.kd * attr.kh * attr.kw + attr.do_relu ? 1 : 0);
}

pool_attr_t top::AvgPoolOp::parseParam() {
  pool_attr_t p = {0};
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());
  if (getKernelShape().size() == 3) {
    p.n = ishape[0];
    p.c = ishape[1];
    p.id = ishape[2];
    p.ih = ishape[3];
    p.iw = ishape[4];
    p.od = oshape[2];
    p.oh = oshape[3];
    p.ow = oshape[4];
    p.kd = kernel->at(0);
    p.kh = kernel->at(1);
    p.kw = kernel->at(2);
    p.sd = stride->at(0);
    p.sh = stride->at(1);
    p.sw = stride->at(2);
    p.pad_d = pad->at(0);
    p.pad_h = pad->at(1);
    p.pad_w = pad->at(2);
    p.pad_d_after = pad->at(3);
    p.pad_h_after = pad->at(4);
    p.pad_w_after = pad->at(5);
  } else if (getKernelShape().size() == 2) {
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.sd = 1;
    module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
    module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
    p.kh = kernel->at(0);
    p.kw = kernel->at(1);
    p.sh = stride->at(0);
    p.sw = stride->at(1);
    p.pad_h = pad->at(0);
    p.pad_w = pad->at(1);
    p.pad_h_after = pad->at(2);
    p.pad_w_after = pad->at(3);
  } else if (getKernelShape().size() == 1) {
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.kw = 1;
    p.sd = 1;
    p.sw = 1;
    module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
    module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
    p.kh = kernel->at(0);
    p.sh = stride->at(0);
    p.pad_h = pad->at(0);
    p.pad_h_after = pad->at(1);
  }
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}

LogicalResult top::AvgPoolOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  auto attr = parseParam();
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
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], module::getNumElements(getOutput()),
                  limit);
  }
  return success();
}
