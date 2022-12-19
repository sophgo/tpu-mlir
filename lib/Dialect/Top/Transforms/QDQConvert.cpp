//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::helper;
namespace tpu_mlir {
namespace top {

// A pattern to convert QuantizeLinearOp return element type to quant.calibrated
// for I don't want to do the conversion using python.
class QuantizeLinearCastTypePattern : public RewritePattern {
public:
  QuantizeLinearCastTypePattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(top::QuantizeLinearOp::getOperationName(), benefit,
                       context) {}

  QuantizeLinearCastTypePattern(MLIRContext *context)
      : RewritePattern(top::QuantizeLinearOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
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
        *Module::getF64Array(op->getAttr("y_scale").dyn_cast<ArrayAttr>());
    auto y_zero_point =
        *Module::getI32Array(op->getAttr("y_zero_point").dyn_cast<ArrayAttr>());
    for (auto i : y_scale)
      std::cout << i << " ";
    std::cout << std::endl;
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
};

// A pattern to fuse quantizelinear with the former ops.
class QuantizeLinearFusePattern : public RewritePattern {
public:
  QuantizeLinearFusePattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(top::QuantizeLinearOp::getOperationName(), benefit,
                       context) {}

  QuantizeLinearFusePattern(MLIRContext *context)
      : RewritePattern(top::QuantizeLinearOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
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
    while (!op->getUses().empty()) {
      op->getUses().begin()->set(oprand);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class RemoveDequantizeLinearPattern : public RewritePattern {
public:
  RemoveDequantizeLinearPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(top::DequantizeLinearOp::getOperationName(), benefit,
                       context) {}

  RemoveDequantizeLinearPattern(MLIRContext *context)
      : RewritePattern(top::DequantizeLinearOp::getOperationName(), 1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value operand = op->getOperand(0);
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (auto weightOp = dyn_cast<WeightOp>(formerOp)) {
      auto scale =
          *Module::getF64Array(dyn_cast<DequantizeLinearOp>(op).x_scale());
      weightOp->setAttr("weight_scale",
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
    while (!op->getUses().empty()) {
      op->getUses().begin()->set(operand);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class QDQConvertPass : public QDQConvertBase<QDQConvertPass> {
public:
  QDQConvertPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    ConversionTarget target(*context);

    target.addIllegalOp<QuantizeLinearOp, DequantizeLinearOp>();

    RewritePatternSet patterns(context);
    patterns.add<QuantizeLinearCastTypePattern>(context);
    patterns.add<QuantizeLinearFusePattern>(context);
    patterns.add<RemoveDequantizeLinearPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    Module::updateModuleTypes(func);
    Module::setState(func, Module::State::TOP_CALIBRATED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createQDQConvertPass() {
  return std::make_unique<QDQConvertPass>();
}

} // namespace top
} // namespace tpu_mlir
