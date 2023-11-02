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

void ConvOp::getAsmResultNames(
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

typedef enum {
  CONV = 0,
  PD   = 1,
  MM   = 2,
  AR   = 3,
  RQDQ = 4,
  TRANS_BC = 5,
  SG   = 6,
  LAR  = 7,
  SFU  = 9,
  LIN  = 10,
  SYS_TRWR = 12,
  CMP  = 13,
  VC   = 14,
  SYS  = 15,
} TSK_TYPE;

typedef enum {
  PAD_CONSTANT    = 0,
  PAD_REFLECTION  = 1,
  PAD_REPLICATION = 2,
  PAD_CIRCULAR    = 3
} PAD_MODE;

typedef enum {
  MM_NORMAL = 1,
  MM_WRQ = 2,
  MM_WRQ_RELU = 3,
  MM_NN = 4,
  MM_NT = 5,
  MM_TT = 6,
} MM_OP;

LogicalResult MatMulOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = TSK_TYPE::MM;
  reg.tsk_eu_typ = MM_OP::MM_NORMAL;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.opt_left_tran = getLeftIsTransposed();
  reg.opt_res_add = getAddResult();
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
  }

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

LogicalResult ConvOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = TSK_TYPE::CONV;
  reg.tsk_eu_typ = 0;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.opt_res_add = getAddResult();
  reg.opt_relu = getDoRelu();

  auto outInfo = getValueInfo(getResult());
  reg.opt_res0_sign = outInfo.getSign();
  reg.res0_n = outInfo.getShape()[0];
  reg.res0_c = outInfo.getShape()[1];
  reg.res0_h = outInfo.getShape()[2];
  reg.res0_w = outInfo.getShape()[3];
  reg.res0_addr = outInfo.getAddr();

  auto inputInfo = getValueInfo(getInput());
  if (inputInfo.getShape()[0] != outInfo.getShape()[0])
    return failure();

  reg.opt_opd0_prec = inputInfo.getPrec();
  reg.opt_opd0_sign = inputInfo.getSign();
  reg.opd0_addr = inputInfo.getAddr();
  {
    auto shape = inputInfo.getShape();
    reg.opd0_c = shape[1];
    reg.opd0_h = shape[2];
    reg.opd0_w = shape[3];
  }

  reg.short_opd0_str = 0;

  auto kerInfo = getValueInfo(getKernel());

  if (inputInfo.getPrec() != kerInfo.getPrec())
    return failure();

  reg.opt_kernel_rotate = false;
  reg.opt_opd1_sign = kerInfo.getSign();
  reg.opd1_addr = kerInfo.getAddr();

  if (kerInfo.isConst()) {
    reg.opt_opd1_const = true;
  } else {
    reg.opt_opd1_const = false;
    auto shape = kerInfo.getShape();
    if (getOc() != shape[1])
      return failure();
    if (getKh() != shape[2])
      return failure();
    if (getKw() != shape[3])
      return failure();
    reg.opd1_h = shape[2];
    reg.opd1_w = shape[3];
  }

  reg.pad_mode = PAD_CONSTANT;
  reg.opd0_up_pad = getPh();
  reg.opd0_dn_pad = getPh();
  reg.opd0_lf_pad = getPw();
  reg.opd0_rt_pad = getPw();

  reg.res_op_x_str = getSw();
  reg.res_op_y_str = getSh();
  reg.opt_opd3_const = false;

  reg.opd0_x_ins0 = 0;
  reg.opd0_y_ins0 = 0;
  reg.opd1_x_ins0 = getDw();
  reg.opd1_y_ins0 = getDh();

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
