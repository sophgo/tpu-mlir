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

LogicalResult tpu::LRNOp::init(InferenceParameter &p) {
  auto alpha_ = alpha().convertToDouble();
  auto beta_ = beta().convertToDouble();
  auto bias_ = bias().convertToDouble();
  auto out_type = Module::getStorageType(output());
  if (out_type.isBF16()) {
    alpha_ = BF16(alpha_);
    beta_ = BF16(beta_);
    bias_ = BF16(bias_);
  } else if (out_type.isF16()) {
    alpha_ = F16(alpha_);
    beta_ = F16(beta_);
    bias_ = F16(bias_);
  }

  auto lrn = new LRN();
  (*lrn)
      .src(p.inputs[0], Module::getShape(input()))
      .dst(p.outputs[0], Module::getShape(output()))
      .size(size())
      .param(alpha_, beta_, bias_)
      .algorithem(algorithm::lrn_across_channels)
      .setup();

  p.handle = (void *)lrn;
  return success();
}

void tpu::LRNOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto lrn = (LRN *)p.handle;
    delete lrn;
    p.handle = nullptr;
  }
}

LogicalResult tpu::LRNOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());

  if (out_type.isa<FloatType>()) {
    auto lrn = (LRN *)p.handle;
    lrn->run();
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else {
    dump();
    llvm_unreachable("not support type");
  }

  return success();
}
