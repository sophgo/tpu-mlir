//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/TypeInterface.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Interfaces/TypeInterface.cpp.inc"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {

bool type_need_cast(Type from, Type to) {
  auto f_sType = module::getStorageType(from);
  auto t_sType = module::getStorageType(to);
  if (f_sType == t_sType) {
    return false;
  }
  if (f_sType.isIntOrIndex() && t_sType.isIntOrIndex()) {
    if (f_sType.getIntOrFloatBitWidth() == t_sType.getIntOrFloatBitWidth()) {
      return false;
    }
    if (f_sType.isInteger(4) && t_sType.isInteger(8)) {
      return false;
    }
  }
  return true;
}

std::string type_string(mlir::Type type) {
  auto t = module::getStorageType(type);
  std::string str;
  if (t.isF32()) {
    str = "f32";
  } else if (t.isF16()) {
    str = "f16";
  } else if (t.isBF16()) {
    str = "bf16";
  } else if (t.isFloat8E5M2()) {
    str = "f8e5m2";
  } else if (t.isFloat8E5M2FNUZ()) {
    str = "f8e5m2";
  } else if (t.isFloat8E4M3FN()) {
    str = "f8e4m3";
  } else if (t.isFloat8E4M3FNUZ()) {
    str = "f8e4m3";
  } else if (t.isIntOrIndex()) {
    if (t.isUnsignedInteger()) {
      str = "ui";
    } else {
      str = "si";
    }
    auto bit = t.getIntOrFloatBitWidth();
    str += std::to_string(bit);
  } else {
    t.dump();
    llvm_unreachable("unknown type");
  }
  return str;
}

static mlir::Type verifyCompatibleType(mlir::Value in, mlir::Type to,
                                       TypeCastMode &mode) {
  auto op = in.getDefiningOp();
  if (op != nullptr && isa<top::WeightOp, top::NoneOp>(op)) {
    return do_nothing(mode);
  }
  // case: equal, do nothing
  auto from_stype = module::getStorageType(in);
  auto to_stype = module::getStorageType(to);
  if (false == type_need_cast(from_stype, to_stype)) {
    return do_nothing(mode);
  }
  // case: quantize
  bool from_isQuant = module::isUniformQuantized(in);
  bool to_isQuant = module::isUniformQuantized(to);
  if (to_isQuant && !from_stype.isIntOrIndex()) {
    mode = TypeCastMode::DO_QUANTIZE;
    return to_stype;
  }

  if (from_isQuant && !to_stype.isIntOrIndex()) {
    mode = TypeCastMode::DO_DEQUANTIZE;
    return to_stype;
  }

  // case to f8
  if (from_stype.isF32() || from_stype.isF16()) {
    if (to_stype.isFloat8E4M3FN() || to_stype.isFloat8E5M2())
      mode = TypeCastMode::DO_QUANTIZE;
    else
      mode = TypeCastMode::DO_CAST;
  } else if (from_stype.isFloat8E4M3FN() || from_stype.isFloat8E5M2()) {
    if (to_stype.isF32() || to_stype.isF16())
      mode = TypeCastMode::DO_DEQUANTIZE;
    else
      llvm_unreachable("unknown type");
  } else {
    // case: other
    mode = TypeCastMode::DO_CAST;
  }
  return to_stype;
}

mlir::Type type_verify_case_same(mlir::Operation *op, uint64_t opd_idx,
                                 TypeCastMode &mode) {
  mlir::Type toType;
  for (auto t : op->getResultTypes()) {
    if (!t.isa<mlir::NoneType>()) {
      toType = t;
      break;
    }
  }
  return type_verify_case_type(op, opd_idx, toType, mode);
}

mlir::Type type_verify_case_type(mlir::Operation *op, uint64_t opd_idx,
                                 mlir::Type type, TypeCastMode &mode) {
  auto num_opds = op->getNumOperands();
  if (opd_idx >= num_opds) {
    llvm_unreachable("opd_idx is illegal.");
  }
  auto in = op->getOperand(opd_idx);
  if (module::getStorageType(in).isIntOrIndex() &&
      module::getStorageType(type).isIntOrIndex()) {
    if (isa<tpu::AddOp, tpu::AddConstOp, tpu::SubOp, tpu::SubConstOp,
            tpu::MulOp, tpu::MulConstOp, tpu::BinaryShiftOp,
            tpu::BinaryConstShiftOp>(op))
      return do_nothing(mode);
  }
  return verifyCompatibleType(in, type, mode);
}

mlir::Type type_verify_case_i32(mlir::Operation *op, uint64_t opd_idx,
                                TypeCastMode &mode) {
  auto in = op->getOperand(opd_idx);
  auto out = op->getResult(0);
  auto is_qtype = module::isUniformQuantized(in);
  auto stype = module::getStorageType(out);
  if (stype.isInteger(32)) {
    if (is_qtype) {
      return do_nothing(mode);
    } else {
      mode = TypeCastMode::DO_QUANTIZE;
      return Builder(op).getI8Type();
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}

mlir::Type type_verify_case_i16_or_i32(mlir::Operation *op, uint64_t opd_idx,
                                       TypeCastMode &mode) {
  auto in = op->getOperand(opd_idx);
  auto out = op->getResult(0);
  auto is_qtype = module::isUniformQuantized(in);
  auto stype = module::getStorageType(out);
  if (stype.isInteger(16) || stype.isInteger(32)) {
    if (is_qtype) {
      return do_nothing(mode);
    } else {
      mode = TypeCastMode::DO_QUANTIZE;
      return Builder(op).getI8Type();
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}

// this is for matmul in f8 mode, that output is fp32 and input needed is fp8
mlir::Type type_verify_case_f32(mlir::Operation *op, uint64_t opd_idx,
                                TypeCastMode &mode, bool isE4 = true) {
  auto in = op->getOperand(opd_idx);
  auto out = op->getResult(0);
  assert(module::getStorageType(out).isF32());
  auto stype = module::getStorageType(in);
  if (isE4 && stype.isFloat8E4M3FN()) {
    return do_nothing(mode);
  } else if (!isE4 && stype.isFloat8E5M2()) {
    return do_nothing(mode);
  } else {
    mode = TypeCastMode::DO_QUANTIZE;
    if (isE4)
      return Builder(op).getFloat8E4M3FNType();
    else
      return Builder(op).getFloat8E5M2Type();
  }
}

}; // namespace tpu_mlir
