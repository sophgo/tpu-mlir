//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "tpu_mlir/Interfaces/TypeInterface.h"

using namespace mlir;

#include "tpu_mlir/Interfaces/TypeInterface.cpp.inc"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace tpu_mlir::helper;
namespace tpu_mlir {

static mlir::Type verifyCompatibleType(mlir::Value from, mlir::Value to,
                                       TypeCastMode &mode) {
  auto op = from.getDefiningOp();
  if (op != nullptr && isa<top::WeightOp, top::NoneOp>(op)) {
    return do_nothing(mode);
  }
  bool from_isQuant = Quant::isUniformQuantized(from);
  bool to_isQuant = Quant::isUniformQuantized(to);
  auto from_stype = Module::getStorageType(from);
  auto to_stype = Module::getStorageType(to);
  bool from_isInt = from_stype.isIntOrIndex();
  bool to_isInt = to_stype.isIntOrIndex();
  // case: equal, do nothing
  if (from_stype == to_stype) {
    return do_nothing(mode);
  }
  // case: quantize
  if (from_isQuant || to_isQuant) {
    if ((from_isQuant && to_isQuant) || (from_isInt && to_isInt)) {
      return do_nothing(mode);
    }
    if (to_isQuant) {
      mode = TypeCastMode::DO_QUANTIZE;
      return to_stype;
    } else {
      mode = TypeCastMode::DO_DEQUANTIZE;
      return to_stype;
    }
  }
  // case: integer and integer
  if (from_isInt && to_isInt) {
    // u8 => i8, i8 => u8, no need to cast
    if (from_stype.getIntOrFloatBitWidth() ==
        to_stype.getIntOrFloatBitWidth()) {
      return do_nothing(mode);
    }
  }

  // case: other
  mode = TypeCastMode::DO_CAST;
  return to_stype;
}

mlir::Type type_verify_case_same(mlir::Operation *op, uint64_t opd_idx,
                                 TypeCastMode &mode) {
  auto num_opds = op->getNumOperands();
  auto out = op->getResult(0);
  if (opd_idx >= num_opds) {
    llvm_unreachable("opd_idx is illegal.");
  }
  auto in = op->getOperand(opd_idx);
  return verifyCompatibleType(in, out, mode);
}

mlir::Type type_verify_case_i32(mlir::Operation *op, uint64_t opd_idx,
                                TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto in = op->getOperand(opd_idx);
    auto out = op->getResult(0);
    auto is_qtype = Quant::isUniformQuantized(in);
    auto stype = Module::getStorageType(out);
    if (stype.isInteger(32)) {
      if (is_qtype) {
        return do_nothing(mode);
      } else {
        mode = TypeCastMode::DO_QUANTIZE;
        return Builder(op).getI8Type();
      }
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}

}; // namespace tpu_mlir
