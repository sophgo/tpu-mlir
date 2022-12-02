//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

namespace tpu_mlir {

struct LoweringConfig {
  static MLIRContext *context;
  static std::string chip;
  static std::string mode;
  static bool isAsymmetric;
  static bool isQuantized;
  static std::map<std::string, llvm::StringRef> quantize_map;
};

template <typename OpTy> class TopLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {
    Operation *op = opTy.getOperation();
    bool isQuantized = LoweringConfig::isQuantized;
    if (isQuantized) {
      LoweringQuantized(rewriter, opTy);
      return success();
    }
    auto real_mode = LoweringConfig::mode;
    auto op_name = Module::getName(op);
    auto chip_name = Module::getChip(op);
    auto iter = LoweringConfig::quantize_map.find(op_name.str());
    if (iter != LoweringConfig::quantize_map.end()) {
      real_mode = iter->second;
    }
    if (real_mode == Quant::Type::INT8) {
      if (op->hasTrait<trait::SupportFuseRelu>() || isa<top::ReluOp>(op)) {
        op->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.0));
      }
      LoweringINT8(rewriter, opTy, LoweringConfig::isAsymmetric);
    } else if (real_mode == Quant::Type::F16) {
      if (Module::isCV18xx(chip_name)) {
        LoweringBF16(rewriter, opTy);
      } else {
        LoweringF16(rewriter, opTy);
      }
    } else if (real_mode == Quant::Type::BF16) {
      LoweringBF16(rewriter, opTy);
    } else {
      LoweringF32(rewriter, opTy);
    }
    return success();
  }

public:
  virtual void LoweringINT8(PatternRewriter &rewriter, OpTy opTy,
                            bool asymmetric) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringBF16(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringF16(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringF32(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringQuantized(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
};

// ================================
// Lowering Helper Apis
// ================================

// Lowering to a new Operation, with the same operands and same attrs, and
// newType
template <typename OpTy>
static void lowering_common(PatternRewriter &rewriter, Operation *from,
                            Type newType) {
  // rewriter.setInsertionPointAfter(from);
  rewriter.replaceOpWithNewOp<OpTy>(from, newType, from->getOperands(),
                                    from->getAttrs());
}

// lowering to a new Operation, with same operands and same attrs, and quantize
// f32 output to int8 output
template <typename OpTy>
static void lowering_common_int8(PatternRewriter &rewriter, Operation *from,
                                 bool asymmetric = false) {
  assert(from->getNumResults() == 1);
  auto newType = Quant::getQuantInt8Type(from->getResult(0), asymmetric);
  lowering_common<OpTy>(rewriter, from, newType);
}

template <typename ElemTy = Float32Type>
static mlir::Type getQuantFloatType(Value v) {
  Type newType = v.getType();
  if (newType.isa<mlir::NoneType>()) {
    return newType;
  }
  auto sType = Module::getStorageType(v);
  if (sType.isa<ElemTy>() == false) {
    auto shape = Module::getShape(v);
    auto ctx = v.getContext();
    if (Quant::isCalibratedType(v)) {
      auto caliType = Quant::getCalibratedType(v);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          ElemTy::get(ctx), caliType.getMin(), caliType.getMax());
      newType = RankedTensorType::get(shape, newCaliType);
    } else {
      newType = RankedTensorType::get(shape, ElemTy::get(ctx));
    }
  }
  return newType;
}

static mlir::Type getQuantBF16Type(Value v) {
  return getQuantFloatType<BFloat16Type>(v);
}

static mlir::Type getQuantF16Type(Value v) {
  return getQuantFloatType<Float16Type>(v);
}

// lowering to f32/f16/bf16
template <typename OpTy, typename ElemTy>
static void lowering_common_float(PatternRewriter &rewriter, Operation *from) {
  assert(from->getNumResults() == 1);
  auto newType = getQuantFloatType<ElemTy>(from->getResult(0));
  lowering_common<OpTy>(rewriter, from, newType);
}

template <typename OpTy>
static void lowering_common_f32(PatternRewriter &rewriter, Operation *from) {
  lowering_common_float<OpTy, Float32Type>(rewriter, from);
}

template <typename OpTy>
static void lowering_common_bf16(PatternRewriter &rewriter, Operation *from) {
  lowering_common_float<OpTy, BFloat16Type>(rewriter, from);
}

template <typename OpTy>
static void lowering_common_f16(PatternRewriter &rewriter, Operation *from) {
  lowering_common_float<OpTy, Float16Type>(rewriter, from);
}

// from int8 to int8, convert one (scale zp) to another (scale zp)
Value do_transfer(Value in, Value out, bool asymmetric);
Value do_transfer_fp(Value in, Value out, bool asymmetric);

// from int8 to int32
Value do_dequant(Location name_loc, Value input, Type to_type,
                 int64_t multiplier, int64_t rshift,
                 tpu::DequantMode mode, int64_t lshift);

// from int8 to int32
Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode);

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode);

template <typename OpTy>
Value do_binary_saclar(Value input, Type to_type, int64_t scalar) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  newType = RankedTensorType::get(Module::getShape(input), to_stype);

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(scalar)));

  std::string new_name =
      Module::getName(input.getDefiningOp()).str() + "_binary";
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp =
      builder.create<OpTy>(name_loc, newType, ValueRange{input}, attrs);
  return newOp.output();
}

Value do_reshape(Value input, RankedTensorType to_type);
Value do_weight_dequant(Value input, Type to_type, int64_t multiplier, int64_t shift,
                        int64_t lshift);
int32_t do_const_dequant(Value input, int64_t multiplier, int64_t shift,
                        int64_t lshift);
} // namespace tpu_mlir
