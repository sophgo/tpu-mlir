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

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/Attention.h"

LogicalResult tpu::TransformerOp::init(InferenceParameter &p) {
  auto attention = new Attention();
  auto in_shape = module::getShape(getInput());
  auto key_shape = module::getShape(getKeys());
  auto queries_shape = module::getShape(getQueriesWeight());
  int batch = in_shape[0];
  int M_q = in_shape[1];
  int M_k = key_shape[1];
  int K = in_shape[2];
  int64_t d = queries_shape[queries_shape.size() - 1];
  auto scale = getScale().convertToDouble();

  attention->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.inputs[3], p.inputs[4],
                   p.inputs[5], p.inputs[6], p.inputs[7], p.inputs[8], p.inputs[9],
                   p.inputs[10], p.inputs[11], p.outputs[0], batch, M_q, M_k, K,
                   d, scale, 0);
  p.handle = (void *)attention;
  return success();
}

void tpu::TransformerOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto attention = (Attention *)p.handle;
    attention->deinit();
    delete attention;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::TransformerOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto attention = (Attention *)p.handle;
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  attention->run();
  if (out_type.isBF16()) {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  } else if (out_type.isF16()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

LogicalResult tpu::TransformerOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }
  return success();
}

// LogicalResult tpu::TransformerOp::AllowDataSplit(int64_t axis,
//                                             group_type_t group_type) {
//   if (axis == 0) {
//     return success();
//   }

//   auto lshape = module::getShape(getInput());
//   if (lshape.size() == 4 && axis == 2 && getHdimIsBatch()) {
//     return success();
//   }

//   return failure();
// }

mlir::Type tpu::TransformerOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0 || opd_idx == 1) {
    return type_verify_case_i32(getOperation(), opd_idx, mode);
  }
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

// void tpu::TransformerOp::assign_fw_param(void *param) {

// }
