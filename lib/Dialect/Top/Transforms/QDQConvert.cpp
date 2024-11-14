//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {

// A pattern to convert QuantizeLinearOp return element type to quant.calibrated
class QuantizeLinearCastTypePattern : public OpRewriterPatternEx3 {
public:
  QuantizeLinearCastTypePattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearCastTypePattern", benefit,
                             top::QuantizeLinearOp::getOperationName()) {}
  QuantizeLinearCastTypePattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearCastTypePattern", 1,
                             top::QuantizeLinearOp::getOperationName()) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    auto return_value = op->getResult(0);
    auto operand_type = op->getOperand(0).getType().cast<RankedTensorType>();
    assert(return_value.getType().dyn_cast<RankedTensorType>() &&
           "return type of QuantizeLinear check failure.");
    assert(operand_type && "operand type of QuantizeLinear check failure.");
    if (return_value.getType()
            .dyn_cast<RankedTensorType>()
            .getElementType()
            .dyn_cast<quant::CalibratedQuantizedType>()) {
      return failure(); // This op is calibrated.
    }
    auto y_scale =
        *module::getF64Array(op->getAttr("y_scale").dyn_cast<ArrayAttr>());
    auto y_zero_point =
        *module::getI32Array(op->getAttr("y_zero_point").dyn_cast<ArrayAttr>());
    assert(y_scale.size() == y_zero_point.size() &&
           "y_scale.size() & y_zero_point.size() must be the same.");
    assert(y_scale.size() == 1 &&
           "Cannot support per chanel quant for activation tensor now.");
    assert(y_scale[0] > 0 && "Scale should be positive.");
    // TO-Do : support asymmetric
    float min = (std::numeric_limits<int8_t>::min() - y_zero_point[0]) *
                y_scale[0],
          max = (std::numeric_limits<int8_t>::max() - y_zero_point[0]) *
                y_scale[0];
    auto quant_type = quant::CalibratedQuantizedType::get(
        operand_type.getElementType(), min, max);
    auto new_type = RankedTensorType::get(operand_type.getShape(), quant_type);
    return_value.setType(new_type);
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

// A pattern to fuse quantizelinear with the former ops.
class QuantizeLinearFusePattern : public OpRewriterPatternEx3 {
public:
  QuantizeLinearFusePattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearFusePattern", benefit,
                             top::QuantizeLinearOp::getOperationName()) {}
  QuantizeLinearFusePattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearFusePattern", 1,
                             top::QuantizeLinearOp::getOperationName()) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    Value oprand = op->getOperand(0);
    auto quantized_type =
        op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!quantized_type) {
      return failure();
    }
    if (!quantized_type.getElementType()
             .dyn_cast<quant::CalibratedQuantizedType>()) {
      return failure(); // Not quantized.
    }
    oprand.setType(quantized_type);
    while (!op->use_empty()) {
      op->getUses().begin()->set(oprand);
    }
    rewriter.eraseOp(op);
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

class RemoveDequantizeLinearPattern : public OpRewriterPatternEx3 {
public:
  RemoveDequantizeLinearPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "RemoveDequantizeLinearPattern", benefit,
                             top::DequantizeLinearOp::getOperationName()) {}
  RemoveDequantizeLinearPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "RemoveDequantizeLinearPattern", 1,
                             top::DequantizeLinearOp::getOperationName()) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    Value operand = op->getOperand(0);
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (auto weightOp = dyn_cast<WeightOp>(formerOp)) {
      auto scale =
          *module::getF64Array(dyn_cast<DequantizeLinearOp>(op).getXScale());
      weightOp->setAttr("scale",
                        rewriter.getF64ArrayAttr(ArrayRef<double>(scale)));
      auto element_type = rewriter.getIntegerType(8, true);
      auto operand_type = operand.getType().cast<RankedTensorType>();
      auto new_type =
          RankedTensorType::get(operand_type.getShape(), element_type);
      weightOp.getResult().setType(new_type);
    } else if (auto quantOp = dyn_cast<QuantizeLinearOp>(formerOp)) {
      // To-Do: Check Type and scale.
    } else if (auto result_type = formerOp->getResult(0)
                                      .getType()
                                      .dyn_cast<RankedTensorType>()) {
      if (!result_type.getElementType()
               .dyn_cast<quant::CalibratedQuantizedType>()) {
        llvm_unreachable("Cannot handle this case.");
      }
    } else {
      llvm_unreachable("Cannot handle this case.");
    }
    while (!op->use_empty()) {
      op->getUses().begin()->set(operand);
    }
    rewriter.eraseOp(op);
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

// Not a graceful pattern. We make the quantized type transparent for Ops like
// reshape, permute, etc...
class CalibratedTypeTransparentPattern : public OpRewriterPatternEx3 {
public:
  CalibratedTypeTransparentPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "CalibratedTypeTransparentPattern",
                             benefit) {}
  CalibratedTypeTransparentPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "CalibratedTypeTransparentPattern", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    if (op->getResults().empty() ||
        isa<top::WeightOp, ReturnOp, top::NoneOp>(op) ||
        isa<tpu::TpuDialect>(op->getDialect())) {
      return failure();
    }
    if (!op->getResult(0).getType().isa_and_nonnull<RankedTensorType>()) {
      return failure();
    }
    auto result_tensor_type =
        op->getResult(0).getType().dyn_cast<RankedTensorType>();
    auto result_quant_type = result_tensor_type.getElementType();
    if (isa<quant::CalibratedQuantizedType, quant::UniformQuantizedType>(
            result_quant_type)) {
      return failure();
    }
    auto succeedingOps = op->getUsers();
    while (!succeedingOps.empty()) {
      auto succeedingOp =
          *succeedingOps.begin(); // We only use the first user of this op. Not
                                  // a good choice.
      if (!succeedingOp->getResult(0)
               .getType()
               .isa_and_nonnull<RankedTensorType>()) {
        return failure();
      }
      auto result_tensor_type =
          succeedingOp->getResult(0).getType().dyn_cast<RankedTensorType>();
      auto result_quant_type = result_tensor_type.getElementType();
      if (isa<quant::CalibratedQuantizedType, quant::UniformQuantizedType>(
              result_quant_type)) {
        op->getResult(0).setType(result_tensor_type);
        return success();
      }
      succeedingOps = succeedingOp->getUsers();
    }
    return failure();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

class QDQConvertPass : public QDQConvertBase<QDQConvertPass> {
public:
  QDQConvertPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    ConversionTarget target(*context);

    target.addIllegalOp<QuantizeLinearOp, DequantizeLinearOp>();

    RewritePatternSet patterns(context), b_patterns(context);
    patterns.add<QuantizeLinearCastTypePattern>(context);
    patterns.add<QuantizeLinearFusePattern>(context);
    patterns.add<RemoveDequantizeLinearPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    b_patterns.add<CalibratedTypeTransparentPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(b_patterns));
    module::updateModuleTypes();
    module::setState(module::State::TOP_CALIBRATED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createQDQConvertPass() {
  return std::make_unique<QDQConvertPass>();
}

} // namespace top
} // namespace tpu_mlir
