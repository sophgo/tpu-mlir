//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTosa/OpLowering.h"
#include "tpu_mlir/Conversion/Conversion.h"
#include "tpu_mlir/Support/Module.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTOSA
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

/*
static unsigned apply_time = 0;
struct ModifyFuncOp : public OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    if(apply_time > 0) return failure();


    // auto sym_name = op.getSymName();
    // auto sym_visi = op.getSymVisibilityAttr();
    // auto arg_attr = op.getArgAttrsAttr();
    // auto res_attr = op.getResAttrsAttr();


    //unsigned operands_size = 0;
    std::vector<Value> new_operands;
    for (auto i : op->getOperands()){
      auto new_type = change_dataformat(i.getType());
      i.setType(new_type);
      new_operands.push_back(i);
    //  operands_size += 1;
    }

    auto funcType = op.getFunctionType();
    std::vector<Type> new_ins;
    std::vector<Type> new_outs;

    for (Type in : funcType.getInputs()) {
      Type new_in = change_dataformat(in);
      new_ins.push_back(new_in);
    }
    for (Type out : funcType.getResults()) {
      Type new_out = change_dataformat(out);
      new_outs.push_back(new_out);
    }

    auto new_funcType = funcType.clone(
                          llvm::makeArrayRef(
                              new_ins.data(), new_ins.size()),
                          llvm::makeArrayRef(
                              new_outs.data(), new_outs.size()));
    //op.setFunctionType(new_funcType);
    //rewriter.replaceOpWithNewOp<mlir::func::FuncOp>(op, sym_name,
    //                  new_funcType, sym_visi, arg_attr, res_attr);
    rewriter.updateRootInPlace(op, [&](){op.setFunctionType(new_funcType);
                                         op->setOperands(new_operands); });
    apply_time += 1;
    return success();
  }

};

struct EraseTopInputOp : public OpRewritePattern<top::InputOp> {
  using OpRewritePattern<top::InputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::InputOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};*/

struct LowerTopWeightOp : public OpRewritePattern<top::WeightOp> {
public:
  LowerTopWeightOp(MLIRContext *ctx, bool include_weight)
      : OpRewritePattern(ctx), include_weight(include_weight) {}

  LogicalResult matchAndRewrite(top::WeightOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() == 1);
    auto outType = change_dataformat(op->getResult(0).getType());
    if (include_weight) {
      auto valptr = op.read_as_float();
      auto new_val = change_weight(valptr, outType);
      auto attr = DenseElementsAttr::get(
          outType, llvm::makeArrayRef(new_val, valptr->size()));
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
    } else {
      // auto out_shape = outType.cast<RankedTensorType>().getShape();
      // auto out_ty = RankedTensorType::get(out_shape, rewriter.getF32Type());
      // attr = DenseElementsAttr::get(out_ty, llvm::ArrayRef<float>());
      auto attr = DenseElementsAttr::get(
          RankedTensorType::get({}, rewriter.getI64Type()),
          llvm::ArrayRef<int64_t>({0}));
      rewriter.replaceOpWithNewOp<mlir::tosa::ConstOp>(op, outType, attr);
    }
    return success();
  }

private:
  bool include_weight;
};

struct ConvertTopToTosa
    : public ::impl::ConvertTopToTosaBase<ConvertTopToTosa> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp();

    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<mlir::tosa::TosaDialect, mlir::func::FuncDialect>();

    // Change data format for FuncOp
    // patterns.add<ModifyFuncOp>(ctx_);
    // applyPatternsAndFoldGreedily(module_, std::move(patterns));

    // Lower TOP Ops
    patterns.add<LowerTopWeightOp>(patterns.getContext(), includeWeight);
    populateTopToTosaConversionPatterns(&patterns);
    auto config = GreedyRewriteConfig();
    config.maxIterations = 0;
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);

    // Erase TOP::InputOp
    // patterns.clear();
    // patterns.add<EraseTopInputOp>(ctx_);
    // applyPatternsAndFoldGreedily(module_, std::move(patterns));

    module::updateModuleTypes();
    module::setState(module::State::TOSA_F32);
  }

protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};

std::unique_ptr<Pass> createConvertTopToTosa() {
  return std::make_unique<ConvertTopToTosa>();
}

} // namespace tpu_mlir
