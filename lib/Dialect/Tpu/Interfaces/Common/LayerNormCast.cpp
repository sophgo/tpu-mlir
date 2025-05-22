//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/CastUtils.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Interfaces/IndexingMapsInterface.h"

LogicalResult tpu::LayerNormCastOp::init(InferenceParameter &p) {
  p.handle = nullptr;
  return success();
}

void tpu::LayerNormCastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LayerNormCastOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Op inference skipped at tpu inference!");
  return success();
}

ArrayAttr tpu::LayerNormCastOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  const int axis = getAxis();
  auto inputMap = AffineMap::getMultiDimIdentityMap(axis, context);
  auto empty = AffineMap::get(axis, 0, context);
  SmallVector<AffineMap> indexingMaps{inputMap};
  for (int i = 1, n = getNumOperands(); i < n; ++i) {
    indexingMaps.push_back(empty);
  }
  indexingMaps.push_back(inputMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::LayerNormCastOp::support_multi_core() { return false; }

mlir::Type tpu::LayerNormCastOp::type_verify(uint64_t opd_idx,
                                             TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = module::getStorageType(getInput());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}
