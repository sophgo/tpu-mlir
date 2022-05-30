
#include "sophgo/Dialect/Top/Transforms/Passes.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <map>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
namespace top {

static void castOpToInt8(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto etype = type.getElementType();
  if (!etype.isa<quant::CalibratedQuantizedType>()) {
    return;
  }
  auto qtype = etype.cast<quant::CalibratedQuantizedType>();
  auto ctx = v.getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(v);
  std::vector<NamedAttribute> attrs;
  auto op = v.getDefiningOp();
  auto module = Module::getModuleOp(op);
  auto chip = Module::getChip(module);
  bool asymmetric = false;
  if (chip == Module::Chip::BM1686) {
    asymmetric = true;
  }
  builder.setInsertionPointAfter(op);
  std::string name = Module::getName(op).str();
  attrs.push_back(
      builder.getNamedAttr("name", builder.getStringAttr(name + "_to_int8")));
  auto castOp =
      builder.create<tpu::CastOp>(op->getLoc(), type, ArrayRef<Value>{operands},
                                  ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(castOp.output(), asymmetric);
  auto new_type =
      RankedTensorType::get(type.getShape(), qtype.getExpressedType());
  v.setType(new_type);
  v.replaceAllUsesExcept(castOp.output(), castOp);
}

static void castOpToExpress(Value v, bool asymmetric = false) {
  if (!Quant::isUniformQuantized(v)) {
    return;
  }
  auto ctx = v.getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(v);
  std::vector<NamedAttribute> attrs;
  auto op = v.getDefiningOp();
  builder.setInsertionPointAfter(op);
  std::string name = Module::getName(op).str();
  std::string qname = name + "_quantized";
  op->setAttr("name", builder.getStringAttr(qname));
  attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
  auto castOp = builder.create<tpu::CastOp>(op->getLoc(), v.getType(),
                                            ArrayRef<Value>{operands},
                                            ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantExpressType(castOp.output());
  v.replaceAllUsesExcept(castOp.output(), castOp);
}

template <typename TyOp>
struct ForwardCalibartion : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.input();
    Value out = op.output();
    if (!Quant::isCalibratedType(in)) {
      return failure();
    }
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    auto in_qtype = Quant::getCalibratedType(in);
    auto out_qtype = Quant::getCalibratedType(out);
    if (in_qtype.getMax() == out_qtype.getMax() &&
        in_qtype.getMin() == out_qtype.getMin()) {
      return failure();
    }
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
    out.setType(new_type);
    return success();
  }
};

template <typename TyOp>
struct BackwardCalibartion : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op->getOperand(0);
    Value out = op.output();
    if (!Quant::isCalibratedType(in)) {
      return failure();
    }
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    if (in.hasOneUse() == false) {
      return failure();
    }

    auto in_qtype = Quant::getCalibratedType(in);
    auto out_qtype = Quant::getCalibratedType(out);
    if (in_qtype.getMax() == out_qtype.getMax() &&
        in_qtype.getMin() == out_qtype.getMin()) {
      return failure();
    }
    auto in_type = in.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
    in.setType(new_type);
    return success();
  }
};

// keep output storage type the same with input storage type
template <typename TyOp>
struct ForwardQuantType : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.input();
    Value out = op.output();
    if (!Quant::isUniformQuantized(in)) {
      return failure();
    }
    if (!Quant::isUniformQuantized(out)) {
      return failure();
    }
    auto in_qtype = Quant::getUniformQuantizedType(in);
    auto out_qtype = Quant::getUniformQuantizedType(out);
    if (in_qtype == out_qtype) {
      return failure();
    }
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
    out.setType(new_type);
    return success();
  }
};

struct LoweringPattern : public RewritePattern {
  LoweringPattern(MLIRContext *context, StringRef mode)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context), mode(mode) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto lowering_op = dyn_cast<sophgo::LoweringInterface>(op);
    if (!lowering_op) {
      return failure();
    }
    auto module = Module::getModuleOp(op);
    auto chip = Module::getChip(module);
    // llvm::errs() << "LoweringPattern mode:" << mode <<" op
    // name:"<<Module::getName(op)<<" chip:"<<chip<<"\n";
    Value newValue;
    if (chip == Module::Chip::BM1684) {
      if (mode == Quant::Type::F32) {
        newValue = lowering_op.lowering_f32_bm1684();
      } else {
        newValue = lowering_op.lowering_int8_bm1684();
      }
    } else if (chip == Module::Chip::BM1686) {
      if (mode == Quant::Type::INT8) {
        newValue = lowering_op.lowering_int8_bm1686();
      } else {
        newValue = lowering_op.lowering_fp(mode);
      }
    }

    rewriter.replaceOp(op, {newValue});
    return success();
  }

protected:
  StringRef mode;
};

class LoweringPass : public LoweringBase<LoweringPass> {
public:
  LoweringPass() {}

  void runOnOperation() override {
    module = getOperation();
    state_ = Module::getState(module);
    llvm::errs() << "default quantize mode:" << this->mode << ", is asymmetric "
                 << this->isAsymmetric << ", chip :" << this->chip
                 << ", state:" << state_ << "\n";

    chip_ = StringRef(chip).upper();
    Module::setChip(module, chip_);
    Module::setAsymmetric(module, isAsymmetric);
    mode_ = StringRef(mode).upper();
    ctx_ = module.getContext();

    if (mode_ == Quant::Type::INT8) {
      lowering_to_int8();
    } else if (mode_ == Quant::Type::F32 || mode_ == Quant::Type::F16 ||
               mode_ == Quant::Type::BF16) {
      lowering_to_fp();
    } else {
      llvm_unreachable("unsupport mode");
    }

    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_LOWERED);
  }

protected:
  void lowering_to_int8() {
    if (state_ != Module::State::TOP_CALIBRATED) {
      llvm_unreachable("Mlir state not support quantize");
    }
    RewritePatternSet patterns(ctx_);
    patterns.add<BackwardCalibartion<top::ReluOp>,
                 BackwardCalibartion<top::MaxPoolOp>>(ctx_);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    patterns.clear();
    patterns.add<
        ForwardCalibartion<top::ReluOp>, ForwardCalibartion<top::MaxPoolOp>,
        ForwardCalibartion<top::AvgPoolOp>, ForwardCalibartion<top::ReshapeOp>>(
        ctx_);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    patterns.clear();
    patterns.add<LoweringPattern>(ctx_, mode_);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    if (chip_ == Module::Chip::BM1686) {
      patterns.clear();
      patterns.add<ForwardQuantType<tpu::AvgPoolOp>,
                   ForwardQuantType<tpu::MaxPoolOp>>(ctx_);
      applyPatternsAndFoldGreedily(module, std::move(patterns));
    }
    // cast input and output to fp32
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](InputOp op) { castOpToInt8(op); });
    }
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](func::ReturnOp op) {
        for (auto opd : op.getOperands()) {
          castOpToExpress(opd);
        }
      });
    }
  }
  void lowering_to_fp() {
    RewritePatternSet patterns(ctx_);
    patterns.add<LoweringPattern>(ctx_, mode_);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
  }

protected:
  ModuleOp module;
  llvm::StringRef state_;
  std::string chip_;
  std::string mode_;
  MLIRContext *ctx_;
};

std::unique_ptr<OperationPass<ModuleOp>> createLoweringPass() {
  return std::make_unique<LoweringPass>();
}
} // namespace top
} // namespace sophgo
