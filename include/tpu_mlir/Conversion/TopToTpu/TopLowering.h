//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
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
  static std::map<Operation *, llvm::StringRef> quantize_map;
};

template <typename OpTy> class TopLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {
    Operation* op = opTy.getOperation();
    if (Quant::isUniformQuantized(op->getResult(0))) {
      LoweringQuantized(rewriter, opTy);
      return success();
    }
    auto real_mode = LoweringConfig::mode;
    auto iter = LoweringConfig::quantize_map.find(op);
    if (iter != LoweringConfig::quantize_map.end()) {
      real_mode = iter->second;
    }
    bool isAsymmetric = LoweringConfig::isAsymmetric;
    if (real_mode == Quant::Type::INT8) {
      if (op->hasTrait<trait::SupportFuseRelu>() || isa<top::ReluOp>(op)) {
        op->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.0));
      }
      LoweringINT8(rewriter, opTy, isAsymmetric);
    } else if (real_mode == Quant::Type::F16) {
      LoweringF16(rewriter, opTy);
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

// lowering to f32/f16/bf16
template <typename OpTy, typename ElemTy = Float32Type>
static void lowering_common_float(PatternRewriter &rewriter, Operation *from) {
  assert(from->getNumResults() == 1);
  auto output = from->getResult(0);
  auto sType = Module::getStorageType(output);
  Type newType = output.getType();
  if (sType.isa<ElemTy>() == false) {
    auto shape = Module::getShape(output);
    auto ctx = from->getContext();
    if (Quant::isCalibratedType(output)) {
      auto caliType = Quant::getCalibratedType(output);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          ElemTy::get(ctx), caliType.getMin(), caliType.getMax());
      newType = RankedTensorType::get(shape, newCaliType);
    } else {
      newType = RankedTensorType::get(shape, ElemTy::get(ctx));
    }
  }
  lowering_common<OpTy>(rewriter, from, newType);
}

// from int8 to int8, convert one (scale zp) to another (scale zp)
Value do_transfer(Value in, Value out, bool asymmetric);
Value do_transfer_fp(Value in, Value out, bool asymmetric);

// from int8 to int32
Value do_dequant(Value input, Type to_type, int64_t multiplier, int64_t rshift,
                 tpu::DequantMode mode, int64_t lshift);

// from int8 to int32
Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode);

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode);

// create lookup table
typedef double (*activate_f)(double);

Value create_lookup_table(Value in, Value out, activate_f func,
                          bool asymmetric);

Value create_lookup_table(Operation *owner, const std::vector<float> &table);

} // namespace tpu_mlir
