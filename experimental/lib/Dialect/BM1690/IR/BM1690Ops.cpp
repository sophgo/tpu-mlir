//===- bm1690Ops.cpp - BM1690 operations  ---------------------------------===//
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
#include "tpu-mlir/Dialect/BM1690/IR/BM1690.h"
#include "llvm/ADT/SmallString.h"

namespace tpu_mlir {
namespace bm1690 {
using namespace mlir;

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

template <typename T>
Attribute dummyPropertiesAsAttribute(::mlir::MLIRContext *ctx, T &property) {
  Builder b{ctx};
  return b.getUnitAttr();
}

typedef enum {
  HW_INT8 = 0,
  HW_FP16 = 1,
  HW_FP32 = 2,
  HW_INT16 = 3,
  HW_INT32 = 4,
  HW_BFP16 = 5,
  HW_INT4 = 6,
  HW_FP8 = 7,
  HW_FP20 = 8,
  HW_UNKNOWN = -1
} HWType;

typedef enum {
  GDMA_INT8 = 0,
  GDMA_FP16 = 1,
  GDMA_FP32 = 2,
  GDMA_INT16 = 3,
  GDMA_INT32 = 4,
  GDMA_BF16 = 5,
  GDMA_FP20 = 6,
  GDMA_UNKNOWN = -1
} GDMA_FORMAT;

typedef enum {
  N_BYTE_ALIGN = 0,
  CONTINUOUS = 1,
  COMPACT = 2,
  FREE = 3,
} Layout;

struct getValueInfo {

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
    return cast<MemRefType>(value.getType()).getShape();
  }

  Type getDtype() {
    if (isConst())
      return value.getType();
    return cast<ShapedType>(value.getType()).getElementType();
  }

  // TODO:
  Layout getLayout() { return {}; }

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
    return HW_UNKNOWN;
  }

  GDMA_FORMAT getGDMAFormat() {
    switch (getPrec()) {
    case HW_FP32:
      return GDMA_FP32;
    case HW_FP16:
      return GDMA_FP16;
    case HW_BFP16:
      return GDMA_BF16;
    case HW_INT32:
      return GDMA_INT32;
    case HW_INT16:
      return GDMA_INT16;
    case HW_INT8:
      return GDMA_INT8;
    case HW_FP20:
      return GDMA_FP20;
    default:
      llvm::errs() << "invalid precision";
      return GDMA_UNKNOWN;
    }
  }

private:
  Value value;
};

static inline int get_gdma_format_type_len(GDMA_FORMAT format) {
  switch (format) {
  case GDMA_INT8:
    return 1;
  case GDMA_FP16:
  case GDMA_BF16:
  case GDMA_INT16:
    return 2;
  case GDMA_FP32:
  case GDMA_INT32:
  case GDMA_FP20:
    return 4;
  default:
    return 0;
  }
}

static inline int get_constant_value(const void *p_val, GDMA_FORMAT format) {
  int constant = 0;
  int type_len = get_gdma_format_type_len(format);
  if (format == GDMA_FP20) {
    type_len = 4;
  }
  memcpy(&constant, p_val, type_len);
  return constant;
}

LogicalResult MatMulOp::verifyAndCodeGen() {
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

  reg.opd0_x_ins0 = getInsertions()[0];
  reg.opd0_y_ins0 = getInsertions()[1];
  reg.opd1_x_ins0 = getDilations()[0] - 1;
  reg.opd1_y_ins0 = getDilations()[1] - 1;

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

LogicalResult binary_op_verify(ShortARRegDef &reg, Operation *op,
                               bool input_exchange = false) {
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::AR();
  reg.cmd_id_dep = cast<TIUIdType>(op->getOperandTypes()[1]).getId();
  reg.tsk_opd_num = 2;
  auto leftInfo = getValueInfo(op->getOperand(input_exchange ? 1 : 0));
  auto rightInfo = getValueInfo(op->getOperand(input_exchange ? 0 : 1));
  auto outInfo = getValueInfo(op->getOpResult(0));

  // left
  reg.opt_opd0_prec = leftInfo.getPrec();
  reg.opt_opd0_sign = leftInfo.getSign();
  reg.opd0_addr = leftInfo.getAddr();
  if (leftInfo.isConst()) {
    reg.opt_opd0_const = true;
  } else {
    reg.opt_opd0_const = false;
    if (leftInfo.getLayout() == Layout::FREE) {
      reg.opd0_str = false;
      auto stride = leftInfo.getStride();
      reg.opd0_n_str = stride[0];
      reg.opd0_c_str = stride[1];
      reg.opd0_h_str = stride[2];
      reg.opd0_w_str = stride[3];
    } else {
      reg.opd0_str = true;
    }
  }

  // right
  reg.opt_opd1_prec = rightInfo.getPrec();
  reg.opt_opd1_sign = rightInfo.getSign();
  reg.opd1_addr = rightInfo.getAddr();
  if (rightInfo.isConst()) {
    reg.opt_opd1_const = true;
  } else {
    reg.opt_opd1_const = false;
    if (rightInfo.getLayout() == Layout::FREE) {
      reg.opd1_str = false;
      auto stride = rightInfo.getStride();
      reg.opd1_n_str = stride[0];
      reg.opd1_c_str = stride[1];
      reg.opd1_h_str = stride[2];
      reg.opd1_w_str = stride[3];
    } else {
      reg.opd1_str = true;
    }
  }

  // out
  reg.opt_res0_prec = outInfo.getPrec();
  reg.res0_addr = outInfo.getAddr();
  auto shape = outInfo.getShape();
  reg.res0_n = shape[0];
  reg.res0_c = shape[1];
  reg.res0_h = shape[2];
  reg.res0_w = shape[3];
  if (outInfo.getLayout() == Layout::FREE) {
    reg.res0_str = false;
    auto stride = outInfo.getStride();
    reg.res0_n_str = stride[0];
    reg.res0_c_str = stride[1];
    reg.res0_h_str = stride[2];
    reg.res0_w_str = stride[3];
  } else {
    reg.res0_str = true;
  }
  return success();
}

LogicalResult logical_binary_op_verify(ShortARRegDef &reg, Operation *op) {

  auto leftInfo = getValueInfo(op->getOperand(0));
  auto rightInfo = getValueInfo(op->getOperand(1));
  auto outInfo = getValueInfo(op->getOpResult(0));
  if (leftInfo.getPrec() != rightInfo.getPrec() ||
      leftInfo.getPrec() != outInfo.getPrec() ||
      leftInfo.getSign() != rightInfo.getSign() ||
      leftInfo.getSign() != outInfo.getSign())
    return failure();

  auto ret = binary_op_verify(reg, op);
  if (ret.failed())
    return failure();
  if (reg.opt_opd0_prec == HWType::HW_FP32) {
    reg.opt_opd0_prec = HWType::HW_INT32;
    reg.opt_opd1_prec = HWType::HW_INT32;
    reg.opt_res0_prec = HWType::HW_INT32;
  } else if (reg.opt_opd0_prec == HWType::HW_FP16 ||
             reg.opt_opd0_prec == HWType::HW_BFP16) {
    reg.opt_opd0_prec = HWType::HW_INT16;
    reg.opt_opd1_prec = HWType::HW_INT16;
    reg.opt_res0_prec = HWType::HW_INT16;
  } else if (reg.opt_opd0_prec == HWType::HW_FP8) {
    reg.opt_opd0_prec = HWType::HW_INT8;
    reg.opt_opd1_prec = HWType::HW_INT8;
    reg.opt_res0_prec = HWType::HW_INT8;
  }
  return success();
}

LogicalResult AndOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::AND;
  return logical_binary_op_verify(reg, getOperation());
}

LogicalResult OrOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::OR;
  return logical_binary_op_verify(reg, getOperation());
}

LogicalResult XorOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::XOR;
  return logical_binary_op_verify(reg, getOperation());
}

LogicalResult compare_binary_op_verify(ShortARRegDef &reg, Operation *op) {

  auto leftInfo = getValueInfo(op->getOperand(0));
  auto rightInfo = getValueInfo(op->getOperand(1));
  auto outInfo = getValueInfo(op->getOpResult(0));
  if (leftInfo.getPrec() != rightInfo.getPrec() ||
      leftInfo.getPrec() != outInfo.getPrec() ||
      leftInfo.getSign() != rightInfo.getSign() ||
      leftInfo.getSign() != outInfo.getSign()) {
    return failure();
  }

  return binary_op_verify(reg, op);
}

LogicalResult MaxOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::MAX;
  return compare_binary_op_verify(reg, getOperation());
}

LogicalResult MinOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::MIN;
  return compare_binary_op_verify(reg, getOperation());
}

LogicalResult binary_op_check(ShortARRegDef &reg, Operation *op) {
  // type check
  auto leftInfo = getValueInfo(op->getOperand(0));
  auto rightInfo = getValueInfo(op->getOperand(1));
  auto outInfo = getValueInfo(op->getOpResult(0));
  if ((leftInfo.getPrec() == HWType::HW_FP32 ||
       leftInfo.getPrec() == HWType::HW_FP16 ||
       leftInfo.getPrec() == HWType::HW_BFP16) &&
      rightInfo.getPrec() != HWType::HW_FP8) {
    if (leftInfo.getPrec() != rightInfo.getPrec() ||
        leftInfo.getPrec() != outInfo.getPrec()) {
      return failure();
    }
  } else if (leftInfo.getPrec() == HWType::HW_FP8) {
    if (rightInfo.getPrec() != HWType::HW_FP8 ||
        rightInfo.getPrec() != HWType::HW_FP16 ||
        rightInfo.getPrec() != HWType::HW_FP32) {
      return failure();
    }
  } else if (rightInfo.getPrec() == HWType::HW_FP8) {
    if (leftInfo.getPrec() != HWType::HW_FP8 ||
        leftInfo.getPrec() != HWType::HW_FP16 ||
        leftInfo.getPrec() != HWType::HW_FP32) {
      return failure();
    }
  } else {
    // int8/int16/int32
    if (!(leftInfo.getDtype().isIntOrIndex() &&
          rightInfo.getDtype().isIntOrIndex() &&
          outInfo.getDtype().isIntOrIndex())) {
      return failure();
    }
  }
  return success();
}

LogicalResult compute_binary_op_verify(ShortARRegDef &reg, Operation *op,
                                       bool input_exchange) {

  auto ret = binary_op_verify(reg, op, input_exchange);
  if (ret.failed())
    return failure();
  // shift
  if (op->getNumOperands() > 2) {
    auto shiftInfo = getValueInfo(op->getOperand(2));
    reg.opt_opd2_prec = shiftInfo.getPrec();
    reg.opt_opd2_sign = shiftInfo.getSign();
    reg.opd2_addr = shiftInfo.getAddr();
    reg.opt_opd2_const = shiftInfo.isConst();
    reg.tsk_opd_num = 3;
    if (shiftInfo.getLayout() != Layout::COMPACT)
      return failure();
    if (shiftInfo.getPrec() == HWType::HW_INT8)
      return failure();
  }
  ret = binary_op_check(reg, op);
  if (ret.failed())
    return failure();

  return binary_op_verify(reg, op);
}

LogicalResult AddOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = getIsSaturation() && getLhs().getType().isIntOrIndex()
                       ? tiuType::AR::ADD_SATU
                       : tiuType::AR::ADD;
  reg.sym_range = getIsSaturation();

  auto leftInfo = getValueInfo(getLhs());
  auto rightInfo = getValueInfo(getRhs());
  auto outInfo = getValueInfo(getResult());
  bool input_exchange = rightInfo.getPrec() == HWType::HW_FP8 &&
                        leftInfo.getPrec() != HWType::HW_FP8;
  auto ret = compute_binary_op_verify(reg, getOperation(), input_exchange);
  if (ret.failed())
    return failure();
  if (getShift())
    reg.opd2_n_str = (int)(getRoundMode());

  // sign_check
  if (leftInfo.getDtype().isIntOrIndex()) {
    if (leftInfo.getDtype().isSignedInteger() ||
        rightInfo.getDtype().isSignedInteger()) {
      if (!outInfo.getDtype().isSignedInteger())
        return failure();
    } else {
      if (!outInfo.getDtype().isUnsignedInteger())
        return failure();
    }
  }
  return success();
}

LogicalResult MulOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = getIsSaturation() && getLhs().getType().isIntOrIndex()
                       ? tiuType::AR::MUL_SATU
                       : tiuType::AR::MUL;
  reg.sym_range = getIsSaturation();

  auto leftInfo = getValueInfo(getLhs());
  auto rightInfo = getValueInfo(getRhs());
  auto outInfo = getValueInfo(getResult());
  bool input_exchange = rightInfo.getPrec() == HWType::HW_FP8 &&
                        leftInfo.getPrec() != HWType::HW_FP8;
  auto ret = compute_binary_op_verify(reg, getOperation(), input_exchange);
  if (ret.failed())
    return failure();
  if (getShift())
    reg.opd2_n_str = (int)(getRoundMode());

  // sign_check
  if (leftInfo.getDtype().isIntOrIndex()) {
    if (leftInfo.getDtype().isSignedInteger() ||
        rightInfo.getDtype().isSignedInteger()) {
      if (!outInfo.getDtype().isSignedInteger())
        return failure();
    } else {
      if (!outInfo.getDtype().isUnsignedInteger())
        return failure();
    }
  }
  return success();
}

LogicalResult SubOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = getIsSaturation() && getLhs().getType().isIntOrIndex()
                       ? tiuType::AR::SUB_SATU
                       : tiuType::AR::SUB;
  reg.sym_range = getIsSaturation();

  auto leftInfo = getValueInfo(getLhs());
  auto rightInfo = getValueInfo(getRhs());
  auto outInfo = getValueInfo(getResult());
  bool input_exchange = leftInfo.getPrec() == HWType::HW_FP8 &&
                        rightInfo.getPrec() != HWType::HW_FP8;
  auto ret = compute_binary_op_verify(reg, getOperation(), input_exchange);
  if (ret.failed())
    return failure();
  if (getShift())
    reg.opd2_n_str = (int)(getRoundMode());

  // sign_check
  if (leftInfo.getDtype().isIntOrIndex()) {
    if (!outInfo.getDtype().isSignedInteger())
      return failure();
  }
  return success();
}

LogicalResult ar_unary_op_verify(ShortARRegDef &reg, Operation *op) {

  auto inInfo = getValueInfo(op->getOperand(0));
  auto outInfo = getValueInfo(op->getOpResult(0));
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::AR();
  reg.cmd_id_dep = cast<TIUIdType>(op->getOperandTypes()[1]).getId();
  reg.tsk_opd_num = 1;

  // in
  reg.opt_opd0_prec = inInfo.getPrec();
  reg.opt_opd0_sign = inInfo.getSign();
  reg.opd0_addr = inInfo.getAddr();
  if (inInfo.isConst()) {
    return failure();
  } else {
    reg.opt_opd0_const = false;
    if (inInfo.getLayout() == Layout::FREE) {
      reg.opd0_str = false;
      auto stride = inInfo.getStride();
      reg.opd0_n_str = stride[0];
      reg.opd0_c_str = stride[1];
      reg.opd0_h_str = stride[2];
      reg.opd0_w_str = stride[3];
    } else {
      reg.opd0_str = true;
    }
  }

  // out
  reg.opt_res0_prec = outInfo.getPrec();
  reg.res0_addr = outInfo.getAddr();
  auto shape = outInfo.getShape();
  reg.res0_n = shape[0];
  reg.res0_c = shape[1];
  reg.res0_h = shape[2];
  reg.res0_w = shape[3];
  if (outInfo.getLayout() == Layout::FREE) {
    reg.res0_str = false;
    auto stride = outInfo.getStride();
    reg.res0_n_str = stride[0];
    reg.res0_c_str = stride[1];
    reg.res0_h_str = stride[2];
    reg.res0_w_str = stride[3];
  } else {
    reg.res0_str = true;
  }
  return success();
}

LogicalResult CopyOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::COPY;

  auto inInfo = getValueInfo(getLhs());
  auto outInfo = getValueInfo(getResult());
  auto ret = ar_unary_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();

  if (inInfo.getPrec() != outInfo.getPrec() ||
      inInfo.getSign() != outInfo.getSign()) {
    return failure();
  }
  return success();
}

LogicalResult AbsOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::ABS;

  auto inInfo = getValueInfo(getLhs());
  auto outInfo = getValueInfo(getResult());
  auto ret = ar_unary_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();

  if (inInfo.getPrec() != outInfo.getPrec() ||
      inInfo.getSign() != outInfo.getSign()) {
    return failure();
  }
  return success();
}

LogicalResult NotOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::NOT;

  auto inInfo = getValueInfo(getLhs());
  auto outInfo = getValueInfo(getResult());
  auto ret = ar_unary_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();

  if (reg.opt_opd0_prec == HWType::HW_FP32) {
    reg.opt_opd0_prec = HWType::HW_INT32;
    reg.opt_res0_prec = HWType::HW_INT32;
  } else if (reg.opt_opd0_prec == HWType::HW_FP16 ||
             reg.opt_opd0_prec == HWType::HW_BFP16) {
    reg.opt_opd0_prec = HWType::HW_INT16;
    reg.opt_res0_prec = HWType::HW_INT16;
  } else if (reg.opt_opd0_prec == HWType::HW_FP8) {
    reg.opt_opd0_prec = HWType::HW_INT8;
    reg.opt_res0_prec = HWType::HW_INT8;
  }
  if (inInfo.getPrec() != outInfo.getPrec()) {
    return failure();
  }
  return success();
}

LogicalResult ar_count_verify(ShortARRegDef &reg, Operation *op) {
  auto outInfo = getValueInfo(op->getOpResult(0));
  auto ret = ar_unary_op_verify(reg, op);
  if (ret.failed())
    return failure();

  if (!outInfo.getDtype().isIntOrIndex())
    return failure();
  if (reg.opt_opd0_prec == HWType::HW_FP32) {
    reg.opt_opd0_prec = HWType::HW_INT32;
  } else if (reg.opt_opd0_prec == HWType::HW_FP16 ||
             reg.opt_opd0_prec == HWType::HW_BFP16) {
    reg.opt_opd0_prec = HWType::HW_INT16;
    if (reg.opt_res0_prec == HWType::HW_INT32) {
      return failure();
    }
  } else if (reg.opt_opd0_prec == HWType::HW_FP8) {
    reg.opt_opd0_prec = HWType::HW_INT8;
    if (reg.opt_res0_prec == HWType::HW_INT32 ||
        reg.opt_res0_prec == HWType::HW_INT16) {
      return failure();
    }
  }
  return success();
}

LogicalResult CLOOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::GET_FIRST_ONE;
  return ar_count_verify(reg, getOperation());
}

LogicalResult CLZOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::AR::GET_FIRST_ZERO;
  return ar_count_verify(reg, getOperation());
}

LogicalResult sfu_op_verify(ShortSFURegDef &reg, Operation *op) {

  auto inInfo = getValueInfo(op->getOperand(0));
  auto outInfo = getValueInfo(op->getOpResult(0));
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::SFU();
  reg.cmd_id_dep = cast<TIUIdType>(op->getOperandTypes()[1]).getId();

  if (inInfo.getLayout() != Layout::N_BYTE_ALIGN ||
      outInfo.getLayout() == Layout::N_BYTE_ALIGN) {
    return failure();
  }
  // in
  reg.opt_opd0_prec = inInfo.getPrec();
  reg.opd0_addr = inInfo.getAddr();
  // out
  reg.opt_res0_prec = outInfo.getPrec();
  reg.res0_addr = outInfo.getAddr();
  auto shape = outInfo.getShape();
  reg.res0_n = shape[0];
  reg.res0_c = shape[1];
  reg.res0_h = shape[2];
  reg.res0_w = shape[3];
  return success();
}

LogicalResult TaylorOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::SFU::TAYLOR_4X;
  auto ret = sfu_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();
  auto tableInfo = getValueInfo(getTable());
  reg.opd1_addr = tableInfo.getAddr();
  reg.opd1_n = getLength();
  return success();
}

LogicalResult NormalOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::SFU::NORM;
  return sfu_op_verify(reg, getOperation());
}

LogicalResult RsqrtOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::SFU::RSQ;
  if (getNumIter() > 4 || getNumIter() < 1)
    return failure();
  reg.opd2_n_str = getNumIter() - 1;
  return sfu_op_verify(reg, getOperation());
}

LogicalResult rqdq_op_verify(ShortRQDQRegDef &reg, Operation *op) {

  auto inInfo = getValueInfo(op->getOperand(0));
  auto outInfo = getValueInfo(op->getOpResult(0));
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::RQDQ();
  reg.cmd_id_dep = cast<TIUIdType>(op->getOperandTypes()[1]).getId();

  // in0
  reg.opt_opd0_prec = inInfo.getPrec();
  reg.opt_opd0_sign = inInfo.getSign();
  reg.opd0_addr = inInfo.getAddr();
  // out
  reg.opt_res0_prec = outInfo.getPrec();
  reg.opt_opd2_sign = outInfo.getSign();
  reg.res0_addr = outInfo.getAddr();
  auto shape = outInfo.getShape();
  reg.res0_n = shape[0];
  reg.res0_c = shape[1];
  reg.res0_h = shape[2];
  reg.res0_w = shape[3];
  // in1
  auto in1Info = getValueInfo(op->getOperand(1));
  reg.opd1_addr = in1Info.getAddr();
  if (in1Info.isConst()) {
    reg.opt_opd1_const = true;
  } else {
    reg.opt_opd1_const = false;
    if (inInfo.getLayout() != Layout::CONTINUOUS) {
      return failure();
    }
  }
  // check
  if (inInfo.getLayout() != Layout::N_BYTE_ALIGN ||
      outInfo.getLayout() == Layout::N_BYTE_ALIGN) {
    return failure();
  }
  // TODO: addr align && start lane
  return success();
}

LogicalResult RequantFpOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::RQDQ::RQ_0;
  reg.sym_range = getIsSaturation();
  reg.opd2_n_str = (int)(getRoundMode0()) + ((int)(getRoundMode1()) << 3);
  auto ret = rqdq_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();
  // ckeck
  if (reg.opt_opd1_const) {
    auto in2Info = getValueInfo(getOperand(2));
    if (!in2Info.getDtype().isF32())
      return failure();
    reg.opd2_addr = in2Info.getAddr();
  }
  auto inInfo = getValueInfo(getOperand(0));
  auto in1Info = getValueInfo(getOperand(1));
  auto outInfo = getValueInfo(getResult());
  if (!inInfo.getDtype().isInteger(8) && !inInfo.getDtype().isInteger(16) &&
      !inInfo.getDtype().isInteger(32))
    return failure();
  if (!outInfo.getDtype().isInteger(4) && !outInfo.getDtype().isInteger(8) &&
      !outInfo.getDtype().isInteger(16))
    return failure();
  if (!in1Info.getDtype().isF32())
    return failure();
  if ((int)(getRoundMode0()) < 0 || (int)(getRoundMode0()) > 4)
    return failure();
  if ((int)(getRoundMode1()) < 0 || (int)(getRoundMode1()) > 4)
    return failure();
  return success();
}
LogicalResult DequantFpOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::RQDQ::DQ_0;
  reg.sym_range = getIsSaturation();
  reg.opd2_n_str = (int)(getRoundMode0());
  auto ret = rqdq_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();
  // ckeck
  if (reg.opt_opd1_const) {
    auto in2Info = getValueInfo(getOperand(2));
    if (!in2Info.getDtype().isSignedInteger(16))
      return failure();
    reg.opd2_addr = in2Info.getAddr();
  }
  auto inInfo = getValueInfo(getOperand(0));
  auto in1Info = getValueInfo(getOperand(1));
  auto outInfo = getValueInfo(getResult());
  if (!inInfo.getDtype().isInteger(4) && !inInfo.getDtype().isInteger(8) &&
      !inInfo.getDtype().isInteger(16))
    return failure();
  if (!outInfo.getDtype().isF32())
    return failure();
  if (!in1Info.getDtype().isF32())
    return failure();
  if ((int)(getRoundMode0()) < 0 || (int)(getRoundMode0()) > 2)
    return failure();
  return success();
}
LogicalResult RequantIntOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::RQDQ::RQ_1;
  reg.sym_range = getIsSaturation();
  reg.opd2_n_str = (int)(getRoundMode());
  auto ret = rqdq_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();
  // ckeck
  if (reg.opt_opd1_const) {
    auto in2Info = getValueInfo(getOperand(2));
    auto in3Info = getValueInfo(getOperand(3));
    if (!in2Info.getDtype().isSignedInteger(8) ||
        !in3Info.getDtype().isSignedInteger(16))
      return failure();
    reg.opd2_addr = in2Info.getAddr() + (in3Info.getAddr() << 16);
  }
  auto inInfo = getValueInfo(getOperand(0));
  auto in1Info = getValueInfo(getOperand(1));
  auto outInfo = getValueInfo(getResult());
  if (!inInfo.getDtype().isInteger(8) && !inInfo.getDtype().isInteger(16) &&
      !inInfo.getDtype().isInteger(32))
    return failure();
  if (!outInfo.getDtype().isInteger(4) && !outInfo.getDtype().isInteger(8) &&
      !outInfo.getDtype().isInteger(16))
    return failure();
  if (!in1Info.getDtype().isSignedInteger(32))
    return failure();
  return success();
}
LogicalResult DequantIntOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::RQDQ::DQ_1;
  reg.sym_range = getIsSaturation();
  reg.opd2_n_str = (int)(getRoundMode());
  auto ret = rqdq_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();
  // ckeck
  if (reg.opt_opd1_const) {
    auto in2Info = getValueInfo(getOperand(2));
    auto in3Info = getValueInfo(getOperand(3));
    if (!in2Info.getDtype().isSignedInteger(8) ||
        !in3Info.getDtype().isSignedInteger(16))
      return failure();
    reg.opd2_addr = in2Info.getAddr() + (in3Info.getAddr() << 16);
  }
  auto inInfo = getValueInfo(getOperand(0));
  auto in1Info = getValueInfo(getOperand(1));
  auto outInfo = getValueInfo(getResult());
  if (!outInfo.getDtype().isInteger(8) && !outInfo.getDtype().isInteger(16) &&
      !outInfo.getDtype().isInteger(32))
    return failure();
  if (!inInfo.getDtype().isInteger(4) && !inInfo.getDtype().isInteger(8) &&
      !inInfo.getDtype().isInteger(16))
    return failure();
  if (!in1Info.getDtype().isSignedInteger(32))
    return failure();
  return success();
}

LogicalResult MaxPoolOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::PD();
  reg.tsk_eu_typ = tiuType::PD::MAX_POOLING;
  reg.cmd_id_dep = getDependency().getType().getId();
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
    reg.opd0_h = shape[2];
    reg.opd0_w = shape[3];
  }

  reg.pad_mode = (int)getPadMode();
  reg.opd0_up_pad = getPads()[0];
  reg.opd0_dn_pad = getPads()[1];
  reg.opd0_lf_pad = getPads()[2];
  reg.opd0_rt_pad = getPads()[3];

  reg.res_op_x_str = getStrides()[0];
  reg.res_op_y_str = getStrides()[1];
  reg.opt_opd3_const = false;

  reg.opd0_x_ins0 = getInsertions()[0];
  reg.opd0_y_ins0 = getInsertions()[1];
  reg.opd1_x_ins0 = getDilations()[0] - 1;
  reg.opd1_y_ins0 = getDilations()[1] - 1;

  return success();
}

LogicalResult MinPoolOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::PD();
  reg.tsk_eu_typ = tiuType::PD::MIN_POOLING;
  reg.cmd_id_dep = getDependency().getType().getId();
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
    reg.opd0_h = shape[2];
    reg.opd0_w = shape[3];
  }

  reg.pad_mode = (int)getPadMode();
  reg.opd0_up_pad = getPads()[0];
  reg.opd0_dn_pad = getPads()[1];
  reg.opd0_lf_pad = getPads()[2];
  reg.opd0_rt_pad = getPads()[3];

  reg.res_op_x_str = getStrides()[0];
  reg.res_op_y_str = getStrides()[1];
  reg.opt_opd3_const = false;

  reg.opd0_x_ins0 = getInsertions()[0];
  reg.opd0_y_ins0 = getInsertions()[1];
  reg.opd1_x_ins0 = getDilations()[0] - 1;
  reg.opd1_y_ins0 = getDilations()[1] - 1;

  return success();
}

LogicalResult AvgPoolOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::PD();
  reg.tsk_eu_typ = tiuType::PD::AVG_POOLING;
  reg.cmd_id_dep = getDependency().getType().getId();
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
    reg.opd0_h = shape[2];
    reg.opd0_w = shape[3];
  }

  float _scale = getScale().convertToFloat();
  unsigned int opd1_addr = *(unsigned int *)&_scale;
  reg.opd1_addr = opd1_addr;

  reg.pad_mode = (int)getPadMode();
  reg.opd0_up_pad = getPads()[0];
  reg.opd0_dn_pad = getPads()[1];
  reg.opd0_lf_pad = getPads()[2];
  reg.opd0_rt_pad = getPads()[3];

  reg.res_op_x_str = getStrides()[0];
  reg.res_op_y_str = getStrides()[1];
  reg.opt_opd3_const = false;

  reg.opd0_x_ins0 = getInsertions()[0];
  reg.opd0_y_ins0 = getInsertions()[1];
  reg.opd1_x_ins0 = getDilations()[0] - 1;
  reg.opd1_y_ins0 = getDilations()[1] - 1;

  return success();
}

LogicalResult DepthwiseOp::verifyAndCodeGen() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::PD();
  reg.tsk_eu_typ = tiuType::PD::DEPTHWISE;
  reg.cmd_id_dep = getDependency().getType().getId();
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
    reg.opd0_h = shape[2];
    reg.opd0_w = shape[3];
  }

  auto kerInfo = getValueInfo(getKernel());

  if (inputInfo.getPrec() != kerInfo.getPrec())
    return failure();

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

  reg.opd0_x_ins0 = getInsertions()[0];
  reg.opd0_y_ins0 = getInsertions()[1];
  reg.opd1_x_ins0 = getDilations()[0] - 1;
  reg.opd1_y_ins0 = getDilations()[1] - 1;

  return success();
}

LogicalResult VectorCorrOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::VC();
  auto op_type = getOpType().str();
  if (op_type == "VC_ADD")
    reg.tsk_eu_typ = tiuType::VC::ADD;
  else if (op_type == "VC_SUB")
    reg.tsk_eu_typ = tiuType::VC::SUB;
  else if (op_type == "VC_MUL")
    reg.tsk_eu_typ = tiuType::VC::MUL;
  else if (op_type == "VC_DIV")
    reg.tsk_eu_typ = tiuType::VC::DIV;
  else if (op_type == "VC_MAX")
    reg.tsk_eu_typ = tiuType::VC::MAX;
  else if (op_type == "VC_MIN")
    reg.tsk_eu_typ = tiuType::VC::MIN;
  else if (op_type == "VC_AND")
    reg.tsk_eu_typ = tiuType::VC::AND;
  else if (op_type == "VC_OR")
    reg.tsk_eu_typ = tiuType::VC::OR;
  else if (op_type == "VC_XOR")
    reg.tsk_eu_typ = tiuType::VC::XOR;
  else if (op_type == "VC_SG")
    reg.tsk_eu_typ = tiuType::VC::SG;
  else if (op_type == "VC_SE")
    reg.tsk_eu_typ = tiuType::VC::SE;
  else if (op_type == "VC_SL")
    reg.tsk_eu_typ = tiuType::VC::SL;
  else if (op_type == "VC_ADD_SATU")
    reg.tsk_eu_typ = tiuType::VC::ADD_SATU;
  else if (op_type == "VC_SUB_SATU")
    reg.tsk_eu_typ = tiuType::VC::SUB_SATU;
  else if (op_type == "VC_MUL_SATU")
    reg.tsk_eu_typ = tiuType::VC::MUL_SATU;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.opd2_n_str = 2;

  auto leftInfo = getValueInfo(getLeft());
  auto rightInfo = getValueInfo(getRight());
  int A_c = leftInfo.getShape()[1];
  int A_w = leftInfo.getShape()[3];
  int B_c = rightInfo.getShape()[1];
  int B_w = rightInfo.getShape()[3];

  auto outputInfo = getValueInfo(getResult());
  reg.res0_addr = outputInfo.getAddr();

  reg.opt_opd0_prec = leftInfo.getPrec();
  reg.opt_opd0_sign = leftInfo.getSign();
  reg.opd0_addr = leftInfo.getAddr();
  reg.res0_c = B_c;
  reg.res0_w = B_w;

  reg.opt_opd1_prec = rightInfo.getPrec();
  reg.opt_opd1_sign = rightInfo.getSign();
  reg.opd1_addr = rightInfo.getAddr();
  reg.opd0_c = A_c;
  reg.opd0_w = A_w;
  reg.opd1_w = 0;

  return success();
}

LogicalResult CWTransOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::TRANS_BC();
  reg.tsk_eu_typ = getIsCwTrans() ? tiuType::TRANS_BC::TRAN_C_W_TRANSPOSE
                                  : tiuType::TRANS_BC::TRAN_W_C_TRANSPOSE;
  reg.cmd_id_dep = getDependency().getType().getId();

  auto inputInfo = getValueInfo(getInput());
  auto outputInfo = getValueInfo(getResult());
  if (inputInfo.getPrec() != outputInfo.getPrec())
    return failure();

  reg.res0_addr = outputInfo.getAddr();
  reg.opt_res0_prec = outputInfo.getPrec();
  reg.res0_n = inputInfo.getShape()[0];
  reg.res0_c = inputInfo.getShape()[1];
  reg.res0_h = inputInfo.getShape()[2];
  reg.res0_w = inputInfo.getShape()[3];

  reg.opd0_addr = inputInfo.getAddr();
  reg.opd0_c = inputInfo.getShape()[3];
  reg.opd0_w = inputInfo.getShape()[1];

  return success();
}

LogicalResult LaneBroadOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::TRANS_BC();
  reg.tsk_eu_typ = tiuType::TRANS_BC::LANE_BROAD;
  reg.cmd_id_dep = getDependency().getType().getId();

  auto inputInfo = getValueInfo(getInput());
  auto outputInfo = getValueInfo(getResult());

  reg.res0_addr = outputInfo.getAddr();
  reg.res0_n = outputInfo.getShape()[0];
  reg.res0_c = outputInfo.getShape()[1];
  reg.res0_h = outputInfo.getShape()[2];
  reg.res0_w = outputInfo.getShape()[3];

  reg.opt_res0_prec = inputInfo.getPrec();
  reg.opd0_addr = inputInfo.getAddr();
  reg.opd0_c = inputInfo.getShape()[1];
  reg.opd0_w = inputInfo.getShape()[3];

  return success();
}

LogicalResult LaneCopyOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::TRANS_BC();
  reg.tsk_eu_typ = tiuType::TRANS_BC::LANE_COPY;
  reg.cmd_id_dep = getDependency().getType().getId();

  auto inputInfo = getValueInfo(getInput());
  auto outputInfo = getValueInfo(getResult());

  reg.res0_addr = outputInfo.getAddr();
  reg.res0_n = outputInfo.getShape()[0];
  reg.res0_c = outputInfo.getShape()[1];
  reg.res0_h = outputInfo.getShape()[2];
  reg.res0_w = outputInfo.getShape()[3];

  reg.opt_res0_prec = inputInfo.getPrec();
  reg.opd0_addr = inputInfo.getAddr();
  reg.opd0_c = inputInfo.getShape()[1];
  reg.opd0_w = inputInfo.getShape()[3];

  return success();
}

LogicalResult StaticBroadOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::TRANS_BC();
  reg.tsk_eu_typ = tiuType::TRANS_BC::STATIC_BROAD;
  reg.cmd_id_dep = getDependency().getType().getId();

  auto inputInfo = getValueInfo(getInput());
  auto outputInfo = getValueInfo(getResult());

  reg.res0_addr = outputInfo.getAddr();
  reg.res0_n = outputInfo.getShape()[0];
  reg.res0_c = getOutputC();
  reg.res0_h = outputInfo.getShape()[2];
  reg.res0_w = inputInfo.getShape()[3];

  reg.opt_res0_prec = inputInfo.getPrec();
  reg.opd0_addr = inputInfo.getAddr();
  reg.opd0_c = getOutputC();
  reg.opd0_w = inputInfo.getShape()[3];

  return success();
}

LogicalResult StaticDistOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::TRANS_BC();
  reg.tsk_eu_typ = tiuType::TRANS_BC::STATIC_DISTRIBUTE;
  reg.cmd_id_dep = getDependency().getType().getId();

  auto inputInfo = getValueInfo(getInput());
  auto outputInfo = getValueInfo(getResult());

  reg.res0_addr = outputInfo.getAddr();
  reg.res0_n = outputInfo.getShape()[0];
  reg.res0_c = getOutputC();
  reg.res0_h = outputInfo.getShape()[2];
  reg.res0_w = 1;

  reg.opt_res0_prec = inputInfo.getPrec();
  reg.opd0_addr = inputInfo.getAddr();
  reg.opd0_c = getOutputC();
  reg.opd0_w = 1;

  return success();
}

LogicalResult TIUSendMsgOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_typ = tiuType::SYS();
  reg.tsk_eu_typ = tiuType::SYS::SEND_MSG;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.imm = ((long long)(getMsgId() & 0x1ff)) |
            ((long long)(getWaitCnt() & 0xff) << 16);
  return success();
}

LogicalResult TIUWaitMsgOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_typ = tiuType::SYS();
  reg.tsk_eu_typ = tiuType::SYS::WAIT_MSG;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.imm = ((long long)(getMsgId() & 0x1ff)) |
            ((long long)(getSendCnt() & 0xff) << 16);
  return success();
}

LogicalResult GDMAMatrixMoveOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_type = dmaType::MATRIX();
  reg.cmd_special_function =
      getTranspose() ? dmaType::MATRIX::TRANS : dmaType::MATRIX::NONE;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.data_format = getValueInfo(getInput()).getGDMAFormat();
  reg.src_start_addr = getValueInfo(getInput()).getAddr();
  reg.dst_start_addr = getValueInfo(getResult()).getAddr();
  reg.src_nsize = 1;
  reg.src_csize = getValueInfo(getInput()).getShape()[1];
  reg.src_hsize = 1;
  reg.src_wsize = getValueInfo(getInput()).getShape()[3];
  reg.dst_nsize = 1;
  reg.dst_csize = getValueInfo(getResult()).getShape()[1];
  reg.dst_hsize = 1;
  reg.dst_wsize = getValueInfo(getResult()).getShape()[3];
  reg.localmem_mask = -1;
  return success();
}

LogicalResult GDMAGeneralMoveOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_type = dmaType::GENERAL();
  reg.cmd_special_function = dmaType::GENERAL::NONE;
  reg.cmd_id_dep = getDependency().getType().getId();
  auto inputInfo = getValueInfo(getInput());
  if (inputInfo.isConst()) {
    reg.fill_constant_en = true;
    reg.constant_value = inputInfo.getConst();
  } else {
    reg.fill_constant_en = false;
  }
  reg.data_format = inputInfo.getGDMAFormat();
  reg.src_cstride = getCount();
  reg.src_start_addr = inputInfo.getAddr();
  reg.dst_start_addr = getValueInfo(getResult()).getAddr();
  reg.localmem_mask = -1;
  return success();
}

LogicalResult GDMAGeneralBroadcastOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_type = dmaType::GENERAL();
  reg.cmd_special_function = dmaType::GENERAL::BROADCAST;
  reg.cmd_id_dep = getDependency().getType().getId();
  auto inputInfo = getValueInfo(getInput());
  if (inputInfo.isConst()) {
    reg.fill_constant_en = true;
    reg.constant_value = inputInfo.getConst();
  } else {
    reg.fill_constant_en = false;
  }
  reg.data_format = inputInfo.getGDMAFormat();
  reg.src_cstride = getCount();
  reg.src_start_addr = inputInfo.getAddr();
  reg.dst_start_addr = getValueInfo(getResult()).getAddr();
  reg.dst_csize = getOutputC();
  reg.localmem_mask = -1;
  return success();
}

LogicalResult GDMAGatherOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_type = dmaType::GATHER();
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.src_format = getValueInfo(getInput()).getGDMAFormat();
  reg.src_start_addr = getValueInfo(getInput()).getAddr();
  reg.dst_start_addr = getValueInfo(getResult()).getAddr();
  reg.src_csize = getValueInfo(getInput()).getShape()[1];
  reg.src_hsize = getValueInfo(getInput()).getShape()[2];
  reg.src_wsize = getValueInfo(getInput()).getShape()[3];
  reg.dst_csize = getValueInfo(getResult()).getShape()[1];
  reg.dst_hsize = getValueInfo(getResult()).getShape()[2];
  reg.dst_wsize = getValueInfo(getResult()).getShape()[3];
  reg.stride_enable = true;
  reg.src_cstride = getValueInfo(getInput()).getStride()[1];
  reg.src_hstride = getValueInfo(getInput()).getStride()[3];
  reg.index_cstride = getValueInfo(getIndex()).getStride()[1];
  reg.index_hstride = getValueInfo(getIndex()).getStride()[3];
  reg.dst_cstride = getValueInfo(getResult()).getStride()[1];
  reg.dst_hstride = getValueInfo(getResult()).getStride()[3];
  reg.constant_value = getConstVal();
  reg.localmem_mask = -1;
  return success();
}

LogicalResult GDMAScatterOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_type = dmaType::SCATTER();
  reg.cmd_special_function = false;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.src_format = getValueInfo(getInput()).getGDMAFormat();
  reg.src_start_addr = getValueInfo(getInput()).getAddr();
  reg.dst_start_addr = getValueInfo(getResult()).getAddr();
  reg.src_csize = getValueInfo(getInput()).getShape()[1];
  reg.src_hsize = getValueInfo(getInput()).getShape()[2];
  reg.src_wsize = getValueInfo(getInput()).getShape()[3];
  reg.dst_csize = getValueInfo(getResult()).getShape()[1];
  reg.dst_hsize = getValueInfo(getResult()).getShape()[2];
  reg.dst_wsize = getValueInfo(getResult()).getShape()[3];
  reg.stride_enable = true;
  reg.src_cstride = getValueInfo(getInput()).getStride()[1];
  reg.src_hstride = getValueInfo(getInput()).getStride()[3];
  reg.index_cstride = getValueInfo(getIndex()).getStride()[1];
  reg.index_hstride = getValueInfo(getIndex()).getStride()[3];
  reg.dst_cstride = getValueInfo(getResult()).getStride()[1];
  reg.dst_hstride = getValueInfo(getResult()).getStride()[3];
  reg.constant_value = getConstVal();
  reg.localmem_mask = -1;
  return success();
}

LogicalResult GDMAScatterAddOp::verify() {
  auto &reg = getProperties().reg;
  reg.cmd_type = dmaType::SCATTER();
  reg.cmd_special_function = true;
  reg.cmd_id_dep = getDependency().getType().getId();
  reg.src_format = getValueInfo(getInput()).getGDMAFormat();
  reg.src_start_addr = getValueInfo(getInput()).getAddr();
  reg.dst_start_addr = getValueInfo(getResult()).getAddr();
  reg.src_csize = getValueInfo(getInput()).getShape()[1];
  reg.src_hsize = getValueInfo(getInput()).getShape()[2];
  reg.src_wsize = getValueInfo(getInput()).getShape()[3];
  reg.dst_csize = getValueInfo(getResult()).getShape()[1];
  reg.dst_hsize = getValueInfo(getResult()).getShape()[2];
  reg.dst_wsize = getValueInfo(getResult()).getShape()[3];
  reg.stride_enable = true;
  reg.src_cstride = getValueInfo(getInput()).getStride()[1];
  reg.src_hstride = getValueInfo(getInput()).getStride()[3];
  reg.index_cstride = getValueInfo(getIndex()).getStride()[1];
  reg.index_hstride = getValueInfo(getIndex()).getStride()[3];
  reg.dst_cstride = getValueInfo(getResult()).getStride()[1];
  reg.dst_hstride = getValueInfo(getResult()).getStride()[3];

  reg.constant_value = getConstVal();
  reg.localmem_mask = -1;
  return success();
}

LogicalResult lin_op_verify(ShortLINRegDef &reg, Operation *op) {
  reg.cmd_short = true;
  reg.tsk_typ = tiuType::LIN();

  auto leftInfo = getValueInfo(op->getOperand(0));
  auto midInfo = getValueInfo(op->getOperand(1));
  auto rightInfo = getValueInfo(op->getOperand(2));
  auto outInfo = getValueInfo(op->getOpResult(0));

  // left
  reg.opd0_sign = leftInfo.getSign();
  reg.opd0_addr = leftInfo.getAddr();
  auto shape_op0 = leftInfo.getShape();
  auto opd0_c = shape_op0[1];

  // mid
  reg.opd1_sign = midInfo.getSign();
  reg.opd1_addr = midInfo.getAddr();
  auto shape_op1 = leftInfo.getShape();
  auto opd1_c = shape_op1[1];
  if (midInfo.isConst()) {
    reg.opt_opd1_const = true;
  } else {
    reg.opt_opd1_const = false;
  }

  // right
  reg.opd2_sign = rightInfo.getSign();
  reg.opd2_addr = rightInfo.getAddr();
  auto shape_op2 = rightInfo.getShape();
  auto opd2_c = shape_op2[1];
  if (rightInfo.isConst()) {
    reg.opt_opd2_const = true;
  } else {
    reg.opt_opd2_const = false;
  }

  // out
  reg.res0_sign = outInfo.getSign();
  reg.opt_res0_prec = outInfo.getPrec();
  reg.res0_addr = outInfo.getAddr();
  auto shape = outInfo.getShape();
  reg.res0_n = shape[0];
  reg.res0_c = shape[1];
  reg.res0_h = shape[2];
  reg.res0_w = shape[3];
  if (opd0_c != opd1_c || opd0_c != opd2_c || opd1_c != opd2_c) {
    return failure();
  }
  return success();
}

LogicalResult MacOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::LIN::MAC;
  reg.cmd_id_dep = getDependency().getType().getId();
  auto ret = lin_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();

  return success();
}

LogicalResult SubSqrOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::LIN::SUB_SQR;
  reg.cmd_id_dep = getDependency().getType().getId();
  auto ret = lin_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();

  return success();
}

LogicalResult AddSqrOp::verify() {
  auto &reg = getProperties().reg;
  reg.tsk_eu_typ = tiuType::LIN::ADD_SQR;
  reg.cmd_id_dep = getDependency().getType().getId();
  auto ret = lin_op_verify(reg, getOperation());
  if (ret.failed())
    return failure();

  return success();
}

} // namespace bm1690
} // namespace tpu_mlir
#define GET_OP_CLASSES
#include "tpu-mlir/Dialect/BM1690/IR/BM1690Ops.cpp.inc"
