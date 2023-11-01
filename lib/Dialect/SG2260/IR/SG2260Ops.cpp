//===- sg2260Ops.cpp - SG2260 operations  ---------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "tpu_mlir/Dialect/SG2260/IR/SG2260.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace tpu_mlir::sg2260;

void MatMulOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "R0");
  auto id = getId().getType().getId();
  llvm::SmallString<32> specialId;
  llvm::raw_svector_ostream specialName(specialId);
  specialName << "tid" << id;
  setNameFn(getId(), specialName.str());
}

template <typename T>
Attribute dummyPropertiesAsAttribute(::mlir::MLIRContext *ctx, T &property) {
  Builder b{ctx};
  return b.getUnitAttr();
}

struct getValueInfo {
  typedef enum {
    HW_INT8 = 0,
    HW_FP16 = 1,
    HW_FP32 = 2,
    HW_INT16 = 3,
    HW_INT32 = 4,
    HW_BFP16 = 5,
    HW_INT4 = 6,
    HW_FP8 = 7,
    DTYPE_UNKNOWN = -1,
  } HWType;

  getValueInfo(Value value) : value(value){};

  bool isConst() { return matchPattern(value, m_Constant()); }

  uint64_t getAddr() { return 0; }
  llvm::ArrayRef<int64_t> getShape() {
    if (isConst())
      return {};
    return cast<ShapedType>(value.getType()).getShape();
  }

  Type getDtype() {
    if (isConst())
      return value.getType();
    return cast<ShapedType>(value.getType()).getElementType();
  }

  bool getSign() { return !getDtype().isUnsignedInteger(); }

  HWType getPrec() {
    auto type = getDtype();
    if (type.isInteger(4))
      return HW_INT4;
    if (type.isInteger(8))
      return HW_INT8;
    if (type.isInteger(16))
      return HW_INT16;
    if (type.isInteger(32))
      return HW_INT32;
    if (type.isF32())
      return HW_FP32;
    if (type.isBF16())
      return HW_BFP16;
    if (type.isF16())
      return HW_FP16;
    type.dump();
    llvm_unreachable("Unsupport type \n");
    return DTYPE_UNKNOWN;
  }

private:
  Value value;
};

LogicalResult MatMulOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = 2;
  reg.tsk_eu_typ = 1;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.opt_left_tran = getLeftIsTransposed();
  reg.opt_res_add = getAddRestult();
  reg.opt_relu = getDoRelu();

  auto leftInfo = getValueInfo(getLeft());
  reg.opt_opd0_prec = leftInfo.getPrec();
  reg.opt_opd0_sign = leftInfo.getSign();
  reg.opd0_addr = leftInfo.getAddr();
  if (leftInfo.isConst()) {
    reg.opt_opd0_const = true;
  } else {
    reg.opt_opd0_const = false;
    auto shape = leftInfo.getShape();
    reg.opd0_n = shape[0];
    reg.opd0_c = shape[1];
    reg.opd0_w = shape[3];
  };

  auto rightInfo = getValueInfo(getRight());

  if (leftInfo.getPrec() != rightInfo.getPrec())
    return failure();

  reg.opt_opd1_sign = rightInfo.getSign();
  reg.opd1_addr = rightInfo.getAddr();

  if (rightInfo.isConst()) {
    return failure();
  }
  reg.opd1_w = rightInfo.getShape()[3];

  auto outInfo = getValueInfo(getResult());
  reg.opt_res0_sign = outInfo.getSign();
  reg.res0_c = outInfo.getShape()[1];
  reg.res0_w = outInfo.getShape()[3];
  reg.res0_addr = outInfo.getAddr();

  if (getBias()) {
    auto biasInfo = getValueInfo(getBias());
    reg.opd2_addr = biasInfo.getAddr();
    reg.opt_opd2_sign = biasInfo.getSign();
    reg.opt_opd2_const = biasInfo.isConst();
  }
  return success();
}

#define GET_OP_CLASSES
#include "tpu_mlir/Dialect/SG2260/IR/SG2260Ops.cpp.inc"
