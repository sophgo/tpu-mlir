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
#include "tpu-mlir/Dialect/SG2260/IR/SG2260.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace tpu_mlir::sg2260;

void getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn,
                       Operation::result_range restults) {
  auto setFun = [&setNameFn](Value value) -> void {
    if (isa<MemRefType>(value.getType()))
      return setNameFn(value, "R0"); // TODO
    if (auto tiuId = dyn_cast<TIUIdType>(value.getType())) {
      auto id = tiuId.getId();
      llvm::SmallString<32> specialId;
      llvm::raw_svector_ostream specialName(specialId);
      specialName << "tiu" << id;
      return setNameFn(value, specialName.str());
    }
    if (auto dmaId = dyn_cast<DMAIdType>(value.getType())) {
      auto id = dmaId.getId();
      llvm::SmallString<32> specialId;
      llvm::raw_svector_ostream specialName(specialId);
      specialName << "dma" << id;
      return setNameFn(value, specialName.str());
    }
  };
  llvm::for_each(restults, setFun);
}

void MatMulOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ::getAsmResultNames(setNameFn, getResults());
}

void ConvOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  ::getAsmResultNames(setNameFn, getResults());
}

void DMATensorOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ::getAsmResultNames(setNameFn, getResults());
}

void DMATensorTransOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ::getAsmResultNames(setNameFn, getResults());
}

void DMATensorBroadcastOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  ::getAsmResultNames(setNameFn, getResults());
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

  uint64_t getConst() { return 0; }

  uint64_t getAddr() { return 0; }

  llvm::ArrayRef<int64_t> getShape() {
    if (isConst())
      return {};
    return cast<ShapedType>(value.getType()).getShape();
  }

  llvm::ArrayRef<int64_t> getStride() {
    if (isConst())
      return {};
    // TODO
    return cast<MemRefType>(value.getType()).getShape();
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
  reg.tsk_typ = tiuType::MM();
  reg.tsk_eu_typ = tiuType::MM::NORMAL;
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
  reg.tsk_typ = tiuType::CONV();
  reg.tsk_eu_typ = tiuType::CONV::NORMAL;
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

  reg.opd0_str = 0;

  auto kerInfo = getValueInfo(getKernel());

  if (inputInfo.getPrec() != kerInfo.getPrec())
    return failure();

  reg.opt_kernel_rotate = false;
  reg.opt_opd1_sign = kerInfo.getSign();
  reg.opd1_addr = kerInfo.getAddr();

  if (kerInfo.isConst()) {
    reg.opt_opd1_const = true;
  } else {
    auto resInfo = getValueInfo(getResult());
    reg.opt_opd1_const = false;
    auto kerShape = kerInfo.getShape();
    auto resShape = resInfo.getShape();
    if (resShape[1] != kerShape[1])
      return failure();
    if (getKernelShape()[0] != kerShape[2])
      return failure();
    if (getKernelShape()[1] != kerShape[2])
      return failure();
    reg.opd1_h = getKernelShape()[0];
    reg.opd1_w = getKernelShape()[1];
  }

  reg.pad_mode = (int)getPadMode();
  reg.opd0_up_pad = getPads()[0];
  reg.opd0_dn_pad = getPads()[1];
  reg.opd0_lf_pad = getPads()[2];
  reg.opd0_rt_pad = getPads()[3];

  reg.res_op_x_str = getStrides()[0];
  reg.res_op_y_str = getStrides()[1];
  reg.opt_opd3_const = false;

  reg.opd0_x_ins0 = getInputInsert0()[0];
  reg.opd0_y_ins0 = getInputInsert0()[1];
  reg.opd1_x_ins0 = getKernelInsert0()[0];
  reg.opd1_y_ins0 = getKernelInsert0()[1];

  if (getBias()) {
    auto biasInfo = getValueInfo(getBias());
    reg.opd2_addr = biasInfo.getAddr();
    reg.opt_opd2_sign = biasInfo.getSign();
    reg.opt_opd2_const = biasInfo.isConst();
  }

  return success();
}

LogicalResult DMATensorBaseVerify(DMATensorRegDef &reg, Operation *op) {
  if (!isa<DMATensorOp, DMATensorTransOp, DMATensorBroadcastOp>(op))
    return failure();

  reg.cmd_short = true;
  reg.cmd_id_dep = cast<TIUIdType>(op->getOperandTypes()[1]).getId();
  reg.cmd_type = dmaType::TENSOR();
  auto srcInfo = getValueInfo(op->getOperand(0));
  reg.src_data_format = srcInfo.getPrec();
  if (srcInfo.isConst()) {
    reg.fill_constant_en = true;
    reg.constant_value = srcInfo.getConst();
  } else {
    auto srcShape = srcInfo.getShape();
    reg.src_nsize = srcShape[0];
    reg.src_csize = srcShape[1];
    reg.src_hsize = srcShape[2];
    reg.src_wsize = srcShape[3];
    auto srcStride = srcInfo.getStride();
    reg.src_nstride = srcStride[0];
    reg.src_cstride = srcStride[1];
    reg.src_hstride = srcStride[2];
    reg.src_wstride = srcStride[3];
  }

  auto dstInfo = getValueInfo(op->getOpResult(0));
  auto desShape = dstInfo.getShape();
  reg.dst_nsize = desShape[0];
  reg.dst_csize = desShape[1];
  reg.dst_hsize = desShape[2];
  reg.dst_wsize = desShape[3];
  auto dstStride = dstInfo.getStride();
  reg.dst_nstride = dstStride[0];
  reg.dst_cstride = dstStride[1];
  reg.dst_hstride = dstStride[2];
  reg.dst_wstride = dstStride[3];
  return success();
}

LogicalResult DMATensorOp::verify() {
  auto ret = DMATensorBaseVerify(getProperties().reg, getOperation());
  if (ret.failed())
    return failure();

  auto reg = getProperties().reg;
  reg.cmd_special_function = dmaType::TENSOR::NONE;
  return success();
}

LogicalResult DMATensorTransOp::verify() {
  auto ret = DMATensorBaseVerify(getProperties().reg, getOperation());
  if (ret.failed())
    return failure();

  auto reg = getProperties().reg;
  reg.cmd_special_function = dmaType::TENSOR::TRANS;
  return success();
}

LogicalResult DMATensorBroadcastOp::verify() {
  auto ret = DMATensorBaseVerify(getProperties().reg, getOperation());
  if (ret.failed())
    return failure();

  auto reg = getProperties().reg;
  reg.cmd_special_function = dmaType::TENSOR::BROADCAST;
  return success();
}

#define GET_OP_CLASSES
#include "tpu-mlir/Dialect/SG2260/IR/SG2260Ops.cpp.inc"
