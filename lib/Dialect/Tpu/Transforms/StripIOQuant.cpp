//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {

bool isF32(Value v) { return isa<Float32Type>(module::getStorageType(v)); }
bool isF16(Value v) {
  return isa<Float16Type>(module::getStorageType(v)) ||
         isa<BFloat16Type>(module::getStorageType(v));
}

bool isF32toF16(tpu::CastOp op) {
  return isF32(op.getInput()) && isF16(op.getOutput());
}

bool isF16toF32(tpu::CastOp op) {
  return isF32(op.getOutput()) && isF16(op.getInput());
}

static inline std::vector<int64_t> string2vec(std::string slist) {
  std::vector<int64_t> outvec;
  std::string idx_str = "";
  for (auto s : slist) {
    int idx;
    if (s == ',') {
      idx = atoi(idx_str.c_str());
      idx_str = "";
      outvec.push_back(idx);
    } else {
      idx_str += s;
    }
  }
  if (idx_str.size())
    outvec.push_back(atoi(idx_str.c_str()));
  return outvec;
}

struct StripInputQuantTpuCastPattern : public OpRewriterPatternEx<tpu::CastOp> {
public:
  StripInputQuantTpuCastPattern(mlir::MLIRContext *context,
                                std::vector<int64_t> quant_input_idx)
      : OpRewriterPatternEx<tpu::CastOp>(context,
                                         "StripInputQuantTpuCastPattern"),
        quant_input_idx(quant_input_idx) {}

  template <typename TyOp>
  bool handleTyOp(tpu::CastOp &op, const std::vector<int64_t> &quant_input_idx,
                  PatternRewriter &rewriter) const {
    if (auto tyOp = op.getInput().template getDefiningOp<TyOp>()) {
      if (!tyOp.getResult().hasOneUse()) {
        return false;
      }
      auto inputOp = tyOp.getInput().template getDefiningOp<top::InputOp>();
      if (!inputOp) {
        return false;
      }
      int idx = module::getIdx(inputOp.getOperand());
      for (int i = 0; i < quant_input_idx.size(); i++) {
        if (quant_input_idx[i] == idx + 1)
          break;
        if (i == quant_input_idx.size() - 1)
          return false;
      }
      if (!inputOp.getOutput().hasOneUse()) {
        return false;
      }
      auto new_ele_type = module::getElementType(op.getResult());
      auto input_new_type =
          RankedTensorType::get(inputOp.getResult()
                                    .getType()
                                    .template cast<RankedTensorType>()
                                    .getShape(),
                                new_ele_type);
      inputOp.getResult().setType(input_new_type);
      auto ty_new_type =
          RankedTensorType::get(tyOp.getResult()
                                    .getType()
                                    .template cast<RankedTensorType>()
                                    .getShape(),
                                new_ele_type);
      tyOp.getResult().setType(ty_new_type);
      rewriter.replaceOp(op, tyOp.getResult());
      return true;
    }

    return false;
  }

  LogicalResult matchAndRewriteImpl(tpu::CastOp op,
                                    PatternRewriter &rewriter) const override {
    if (auto inputOp = op.getInput().getDefiningOp<top::InputOp>()) {
      int idx = module::getIdx(inputOp.getOperand());
      for (int i = 0; i < quant_input_idx.size(); i++) {
        if (quant_input_idx[i] == idx + 1)
          break;
        if (i == quant_input_idx.size() - 1)
          return failure();
      }
      auto out = inputOp.getOutput();
      if (!out.hasOneUse()) {
        return failure();
      }
      if (!module::isUniformQuantized(op.getOutput()) && !isF32toF16(op)) {
        // special case for 18xx MatchTemplateOp
        if (module::getStorageType(op.getOutput()).isUnsignedInteger(8)) {
          auto nextOp = *op->user_begin();
          if (!module::isCV18xx() || !isa<tpu::MatchTemplateOp>(nextOp)) {
            return failure();
          }
        } else {
          return failure();
        }
      }
      out.setType(op.getResult().getType());
      rewriter.replaceOp(op, out);
      return success();
    }

    // input -> reshape -> cast -> any op
    if (handleTyOp<tpu::ReshapeOp>(op, quant_input_idx, rewriter)) {
      return success();
    }

    // for case input -> unsqueeze -> cast
    if (handleTyOp<tpu::UnsqueezeOp>(op, quant_input_idx, rewriter)) {
      return success();
    }

    return failure();
  };
  bool shouldPrint(tpu::CastOp op) const override { return false; }

private:
  std::vector<int64_t> quant_input_idx;
};

struct StripInputQuantCpuCastPattern
    : public OpRewriterPatternEx<tpu::GenericCpuOp> {
public:
  StripInputQuantCpuCastPattern(mlir::MLIRContext *context,
                                std::vector<int64_t> quant_input_idx)
      : OpRewriterPatternEx<tpu::GenericCpuOp>(context,
                                               "StripInputQuantCpuCastPattern"),
        quant_input_idx(quant_input_idx) {}

  LogicalResult matchAndRewriteImpl(tpu::GenericCpuOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getCpuOpName() != "quant") {
      return failure();
    }
    if (auto inputOp = op.getInputs()[0].getDefiningOp<top::InputOp>()) {
      if (!inputOp.getResult().hasOneUse())
        return failure();
      int idx = module::getIdx(inputOp.getOperand());
      for (int i = 0; i < quant_input_idx.size(); i++) {
        if (quant_input_idx[i] == idx + 1)
          break;
        if (i == quant_input_idx.size() - 1)
          return failure();
      }
      inputOp->getResult(0).setType(op.getResults()[0].getType());
      rewriter.replaceOp(op, inputOp.getResult());
      return success();
    }
    return failure();
  };
  bool shouldPrint(tpu::GenericCpuOp op) const override { return false; }

private:
  std::vector<int64_t> quant_input_idx;
};

class StripOutputQuantTpuCastPattern : public OpRewriterPatternEx<tpu::CastOp> {
public:
  StripOutputQuantTpuCastPattern(mlir::MLIRContext *context,
                                 std::vector<int64_t> quant_output_idx)
      : OpRewriterPatternEx<tpu::CastOp>(context,
                                         "StripOutputQuantTpuCastPattern"),
        quant_output_idx(quant_output_idx) {}

  LogicalResult matchAndRewriteImpl(tpu::CastOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getOutput().hasOneUse() &&
        isa<ReturnOp>(op.getOutput().use_begin().getUser())) {
      auto ReturnOp = op.getOutput().use_begin().getUser();
      for (int i = 0; i < quant_output_idx.size(); i++) {
        if (ReturnOp->getOperand(quant_output_idx[i] - 1) == op.getOutput())
          break;
        if (i == quant_output_idx.size() - 1)
          return failure();
      }
      auto in = op.getInput();
      if (!module::isUniformQuantized(in) && !isF16toF32(op)) {
        return failure();
      }
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  };
  bool shouldPrint(tpu::CastOp op) const override { return false; }

private:
  std::vector<int64_t> quant_output_idx;
};

class ForceOutputQuantBF16Pattern : public OpRewriterPatternEx<ReturnOp> {
public:
  ForceOutputQuantBF16Pattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReturnOp>(context, "ForceOutputQuantBF16Pattern") {}

  LogicalResult matchAndRewriteImpl(ReturnOp op,
                                    PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto num_opd = op.getNumOperands();
    bool is_fixed = false;
    for (int i = 0; i < num_opd; i++) {
      auto opd = op.getOperand(i);
      auto type = module::getStorageType(opd);
      if (!type.isF32()) {
        continue;
      }
      auto cast_loc = module::getLocLike(opd, "bf16");
      auto cast_shape = module::getShape(opd);
      auto cast_type =
          RankedTensorType::get(cast_shape, rewriter.getBF16Type());
      auto cast_op =
          rewriter.create<tpu::CastOp>(cast_loc, cast_type, ValueRange{opd});
      op.setOperand(i, cast_op.getOutput());
      is_fixed = true;
    }
    return is_fixed ? success() : failure();
  };
  bool shouldPrint(ReturnOp op) const override { return false; }
};

struct StripOutputQuantCpuCastPattern
    : public OpRewriterPatternEx<tpu::GenericCpuOp> {
public:
  StripOutputQuantCpuCastPattern(mlir::MLIRContext *context,
                                 std::vector<int64_t> quant_output_idx)
      : OpRewriterPatternEx<tpu::GenericCpuOp>(
            context, "StripOutputQuantCpuCastPattern"),
        quant_output_idx(quant_output_idx) {}

  LogicalResult matchAndRewriteImpl(tpu::GenericCpuOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isCV18xx()) {
      if (op.getCpuOpName() != "quant") {
        return failure();
      }
      if (op.getOutputs()[0].hasOneUse() &&
          isa<ReturnOp>(op.getOutputs()[0].use_begin().getUser())) {
        auto ReturnOp = op.getOutputs()[0].use_begin().getUser();
        for (int i = 0; i < quant_output_idx.size(); i++) {
          if (ReturnOp->getOperand(quant_output_idx[i] - 1) ==
              op.getOutputs()[0])
            break;
          if (i == quant_output_idx.size() - 1)
            return failure();
        }
        rewriter.replaceOp(op, op.getInputs()[0]);
        return success();
      }
    }
    return failure();
  };
  bool shouldPrint(tpu::GenericCpuOp op) const override { return false; }

private:
  std::vector<int64_t> quant_output_idx;
};

class StripIOQuantPass : public StripIOQuantBase<StripIOQuantPass> {
public:
  StripIOQuantPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto func = module::getMainFuncOp(mOp);
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    if (quant_input) {
      std::vector<int64_t> quant_input_idx = string2vec(quant_input_list);
      patterns.add<StripInputQuantTpuCastPattern>(ctx, quant_input_idx);
      patterns.add<StripInputQuantCpuCastPattern>(ctx, quant_input_idx);
    }
    if (quant_output) {
      std::vector<int64_t> quant_output_idx = string2vec(quant_output_list);
      patterns.add<StripOutputQuantTpuCastPattern>(ctx, quant_output_idx);
      patterns.add<StripOutputQuantCpuCastPattern>(ctx, quant_output_idx);
    } else if (quant_output_bf16) {
      patterns.add<ForceOutputQuantBF16Pattern>(ctx);
    }
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStripIOQuant() {
  return std::make_unique<StripIOQuantPass>();
}
} // namespace tpu
} // namespace tpu_mlir
