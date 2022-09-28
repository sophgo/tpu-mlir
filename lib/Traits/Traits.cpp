//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

#include "mlir/Dialect/Quant/QuantTypes.h"

using namespace mlir;
using namespace tpu_mlir::helper;

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

LogicalResult verifyInOutSameTypeTrait(Operation *op) {
  auto num_opds = op->getNumOperands();
  auto out = op->getResult(0);
  bool out_isQuant = Quant::isUniformQuantized(out);
  auto out_stype = Module::getStorageType(out);
  bool out_isInt = out_stype.isIntOrIndex();
  for (uint32_t i = 0; i < num_opds; i++) {
    auto in_opd = op->getOperand(i);
    auto in_op = in_opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(op)) {
      continue;
    }
    bool in_isQuant = Quant::isUniformQuantized(in_opd);
    auto in_stype = Module::getStorageType(in_opd);
    bool in_isInt = in_stype.isIntOrIndex();
    if (in_stype == out_stype) {
      continue;
    }
    if (out_isQuant && in_isQuant) {
      continue;
    }
    if (out_isInt && in_isInt) {
      if (in_stype.getIntOrFloatBitWidth() ==
          out_stype.getIntOrFloatBitWidth()) {
        continue;
      }
    }
    op->dump();
    return op->emitError("expected input and output with same type");
  }
  return mlir::success();
}

} // namespace impl
} // namespace trait

} // namespace tpu_mlir
