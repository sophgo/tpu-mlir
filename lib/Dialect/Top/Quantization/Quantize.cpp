//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/Transforms/Passes.h"
#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <regex>
#include <map>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
namespace top {

static void castOpToInt8(Value v) {
  if (!Quant::isQuantizedType<quant::CalibratedQuantizedType>(v)) {
    return;
  }

  auto ctx = v.getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(v);
  std::vector<NamedAttribute> attrs;
  auto op = v.getDefiningOp();
  auto module = Module::getModuleOp(op);
  auto chip = Module::getChip(module);
  bool asymmetric = false;
  bool sign = true;
  if (chip == Module::Chip::BM1686) {
    asymmetric = true;
    sign = false;
  }
  builder.setInsertionPointAfter(op);
  std::string name = Module::getName(op).str();
  attrs.push_back(
      builder.getNamedAttr("name", builder.getStringAttr(name + "_to_int8")));
  auto castOp = builder.create<tpu::CastOp>(op->getLoc(), v.getType(),
                                            ArrayRef<Value>{operands},
                                            ArrayRef<NamedAttribute>{attrs});
  Quant::setQuantInt8Type(castOp.output(), asymmetric, sign);
  v.replaceAllUsesExcept(castOp.output(), castOp);
}

static void castOpToExpress(Value v, bool asymmetric = false) {
  if (!Quant::isQuantizedType<quant::UniformQuantizedType>(v)) {
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
    if (!Quant::isQuantizedType<quant::CalibratedQuantizedType>(in)) {
      return failure();
    }
    if (!Quant::isQuantizedType<quant::CalibratedQuantizedType>(out)) {
      return failure();
    }
    auto in_qtype = Quant::getQuantizedType<quant::CalibratedQuantizedType>(in);
    auto out_qtype =
        Quant::getQuantizedType<quant::CalibratedQuantizedType>(out);
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
    if (!Quant::isQuantizedType<quant::CalibratedQuantizedType>(in)) {
      return failure();
    }
    if (!Quant::isQuantizedType<quant::CalibratedQuantizedType>(out)) {
      return failure();
    }
    if (in.hasOneUse() == false) {
      return failure();
    }

    auto in_qtype = Quant::getQuantizedType<quant::CalibratedQuantizedType>(in);
    auto out_qtype =
        Quant::getQuantizedType<quant::CalibratedQuantizedType>(out);
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
    if (!Quant::isQuantizedType<quant::UniformQuantizedType>(in)) {
      return failure();
    }
    if (!Quant::isQuantizedType<quant::UniformQuantizedType>(out)) {
      return failure();
    }
    auto in_qtype = Quant::getQuantizedType<quant::UniformQuantizedType>(in);
    auto out_qtype = Quant::getQuantizedType<quant::UniformQuantizedType>(out);
    if (in_qtype == out_qtype) {
      return failure();
    }
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
    out.setType(new_type);
    return success();
  }
};

struct QuantizationPattern : public RewritePattern {
  QuantizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto quantize_op = dyn_cast<sophgo::QuantizeInterface>(op);
    if (!quantize_op) {
      return failure();
    }
    auto module = Module::getModuleOp(op);
    auto chip = Module::getChip(module);
    Value newValue;
    if (chip == Module::Chip::BM1684) {
      newValue = quantize_op.quantize_int8_bm1684();
    } else if (chip == Module::Chip::BM1686) {
      newValue = quantize_op.quantize_int8_bm1686();
    }
    rewriter.replaceOp(op, {newValue});
    return success();
  }
};

class QuantizePass : public QuantizeBase<QuantizePass> {
public:
  QuantizePass() {}
  void runOnOperation() override {
    llvm::errs() << "default quantize mode:" << this->mode << ", is asymmetric "
                 << this->isAsymmetric << ", chip :" << this->chip << "\n";
    auto module = getOperation();
    auto state = Module::getState(module);
    if (state != Module::State::TOP_CALIBRATED || mode != Quant::Type::INT8) {
      module.dump();
      llvm_unreachable("Mlir state not support quantize");
    }
    StringRef chip_ = StringRef(chip).upper();
    Module::setChip(module, chip_);
    auto ctx = module.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<BackwardCalibartion<top::ReluOp>,
                 BackwardCalibartion<top::MaxPoolOp>>(ctx);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    patterns.clear();
    patterns.add<
        ForwardCalibartion<top::ReluOp>, ForwardCalibartion<top::MaxPoolOp>,
        ForwardCalibartion<top::AvgPoolOp>, ForwardCalibartion<top::ReshapeOp>>(
        ctx);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    patterns.clear();
    patterns.add<QuantizationPattern>(ctx);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    // sync sign type for special op
    if (chip_ == Module::Chip::BM1686) {
      patterns.clear();
      patterns.add<ForwardQuantType<tpu::AvgPoolOp>,
                  ForwardQuantType<tpu::MaxPoolOp>>(ctx);
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
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_QUANTIZED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createQuantizePass() {
  return std::make_unique<QuantizePass>();
}
} // namespace top
} // namespace sophgo
