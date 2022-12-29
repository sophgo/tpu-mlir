//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/Dialect/Quant/QuantTypes.h"


namespace tpu_mlir {

namespace trait {
namespace impl {

static LogicalResult check_type(Value v) {
  auto type = v.getType();
  if (type.isa<NoneType>()) {
    return success();
  }
  if (auto tensor_type = type.dyn_cast<RankedTensorType>()) {
    auto etype = tensor_type.getElementType();
    if (etype.isIntOrFloat()) {
      return success();
    }
    if (etype.isa<quant::UniformQuantizedType,
                  quant::CalibratedQuantizedType>()) {
      return success();
    }
  }
  return failure();
}

LogicalResult verifyTpuTypeRestrictTrait(Operation *op) {
  for (auto out : op->getResults()) {
    if (failed(check_type(out))) {
      return op->emitError("expected tpu supported type");
    }
  }
  return mlir::success();
}

LogicalResult verifyInOutSameShapeTrait(Operation *op) {
  auto in_shape =
      op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto out_shape =
      op->getResult(0).getType().cast<RankedTensorType>().getShape();
  if (in_shape != out_shape) {
    return op->emitError("expected input and output with same shape");
  }
  return mlir::success();
}

LogicalResult verifyInOutSameDimTrait(Operation *op) {
  auto in_shape_size =
      op->getOperand(0).getType().cast<RankedTensorType>().getShape().size();
  auto out_shape_size =
      op->getResult(0).getType().cast<RankedTensorType>().getShape().size();
  if (in_shape_size != out_shape_size) {
    return op->emitError("expected input and output with same shape");
  }
  return mlir::success();
}

} // namespace impl
} // namespace trait

} // namespace tpu_mlir
