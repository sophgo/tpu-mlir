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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::LRNOp::init(InferenceParameter &p) {
  auto alpha_ = getAlpha().convertToDouble();
  auto beta_ = getBeta().convertToDouble();
  auto bias_ = getBias().convertToDouble();
  auto out_type = module::getStorageType(getOutput());
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
      .src(p.inputs[0], module::getShape(getInput()))
      .dst(p.outputs[0], module::getShape(getOutput()))
      .size(getSize())
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
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());

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

LogicalResult tpu::LRNOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return success();
  }
  return failure();
}
