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
      if (module::isCalibratedType(op.getOutput())) {
        auto orig_type = op.getResult().getType().cast<RankedTensorType>();
        auto min = module::getCalibratedType(orig_type).getMin();
        auto max = module::getCalibratedType(orig_type).getMax();
        auto qtype = quant::CalibratedQuantizedType::get(
            orig_type.getElementType(), min, max);
        auto input_new_type =
            RankedTensorType::get(orig_type.getShape(), qtype);
        inputOp.getResult().setType(input_new_type);

        auto ty_orig_type =
            tyOp.getResult().getType().template cast<RankedTensorType>();
        min = module::getCalibratedType(orig_type).getMin();
        max = module::getCalibratedType(orig_type).getMax();
        auto ty_qtype = quant::CalibratedQuantizedType::get(
            ty_orig_type.getElementType(), min, max);
        auto ty_new_type =
            RankedTensorType::get(ty_orig_type.getShape(), ty_qtype);
        tyOp.getResult().setType(ty_new_type);
        rewriter.replaceOp(op, tyOp.getResult());
      } else {
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
      }
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
      if (!module::isCalibratedType(op.getResult()) &&
          !module::isUniformQuantized(op.getOutput())) {
        if (module::isCalibratedType(out)) {
          auto in_type = out.getType().cast<RankedTensorType>();
          auto min = module::getCalibratedType(in_type).getMin();
          auto max = module::getCalibratedType(in_type).getMax();
          auto qtype = quant::CalibratedQuantizedType::get(
              module::getStorageType(op.getResult()), min, max);
          auto newOutputType = RankedTensorType::get(in_type.getShape(), qtype);
          out.setType(newOutputType);
        } else {
          out.setType(op.getResult().getType());
        }
      } else {
        out.setType(op.getResult().getType());
      }
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

struct ForceInputQuantInt8Pattern : public OpRewriterPatternEx<top::InputOp> {
public:
  ForceInputQuantInt8Pattern(mlir::MLIRContext *context,
                             std::vector<int64_t> quant_input_idx)
      : OpRewriterPatternEx<top::InputOp>(context,
                                          "ForceInputQuantInt8Pattern"),
        quant_input_idx(quant_input_idx) {}

  LogicalResult matchAndRewriteImpl(top::InputOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isUniformQuantized(op.getResult()) ||
        module::getStorageType(op.getResult()).isInteger(32) ||
        module::getStorageType(op.getResult()).isInteger(16)) {
      // don't handle already quantied input and integer inputs
      return failure();
    }
    for (int i = 0; i < quant_input_idx.size(); i++) {
      int idx = module::getIdx(op.getOperand());
      if (quant_input_idx[i] == idx + 1)
        break;
      if (i == quant_input_idx.size() - 1)
        return failure();
    }
    // now to change output of input to int8/uint8 and insert cast after
    // it or after reshape/tile/unsqueeze/squeeze
    auto quant_output = [&](Operation *op) {
      auto orig_type = op->getResult(0).getType().cast<RankedTensorType>();
      auto cali_type = module::getCalibratedType(orig_type);
      auto min = cali_type.getMin();
      double scale;
      int64_t zeropoint = 0;
      module::getScaleAndZeroPoint(op->getResult(0), scale, zeropoint,
                                   module::isAsymmetric());
      int64_t qmin = -128, qmax = 127;
      uint32_t flag = quant::QuantizationFlags::Signed;
      if (min >= 0) {
        qmin = 0;
        qmax = 255;
        flag = 0;
      }
      auto qtype = quant::UniformQuantizedType::get(
          flag, IntegerType::get(rewriter.getContext(), 8),
          cali_type.getExpressedType(), scale, zeropoint, qmin, qmax);
      Type newOutputType;
      if (auto shapedType = orig_type.dyn_cast<mlir::ShapedType>()) {
        newOutputType = shapedType.clone(shapedType.getShape(), qtype);
      } else {
        newOutputType = qtype;
      }
      // 4. Directly change the InputOp's result type
      op->getResult(0).setType(newOutputType);
      return orig_type;
    };

    auto orig_type = quant_output(op.getOperation());

    // auto nxt_op = *op->getResult(0).user_begin();
    auto nxt_op = op.getOperation();
    while (isa<tpu::ReshapeOp, tpu::SqueezeOp, tpu::UnsqueezeOp, tpu::TileOp>(
               nxt_op) &&
           nxt_op->getResult(0).hasOneUse()) {
      orig_type = quant_output(nxt_op);
      nxt_op = *nxt_op->getResult(0).user_begin();
    }
    // insert cast after nxt_op
    rewriter.setInsertionPointAfter(nxt_op);
    auto cast_loc = module::getLocLike(nxt_op->getResult(0), "cast_int8");
    auto cast_type = RankedTensorType::get(
        module::getShape(nxt_op->getResult(0)), orig_type.getElementType());
    auto cast_op = rewriter.create<tpu::CastOp>(
        cast_loc, cast_type, ValueRange{nxt_op->getResult(0)});
    nxt_op->getResult(0).replaceAllUsesExcept(cast_op.getOutput(), cast_op);
    return success();
  };
  bool shouldPrint(top::InputOp op) const override { return false; }

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

struct ForceOutputQuantInt8Pattern : public OpRewriterPatternEx<ReturnOp> {
public:
  ForceOutputQuantInt8Pattern(mlir::MLIRContext *context,
                              std::vector<int64_t> quant_output_idx)
      : OpRewriterPatternEx<ReturnOp>(context, "ForceOutputQuantInt8Pattern"),
        quant_output_idx(quant_output_idx) {}

  LogicalResult matchAndRewriteImpl(ReturnOp op,
                                    PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto num_opd = op.getNumOperands();
    bool modified = false;
    for (int i = 0; i < num_opd; i++) {
      auto opd = op.getOperand(i);
      if (module::isUniformQuantized(opd) ||
          module::getStorageType(opd).isInteger(32) ||
          module::getStorageType(opd).isInteger(16)) {
        // don't handle already quantied output and integer outputs
        continue;
      }
      bool do_quant = true;
      for (int j = 0; j < quant_output_idx.size(); j++) {
        if (quant_output_idx[j] == i + 1)
          break;
        if (j == quant_output_idx.size() - 1) {
          do_quant = false;
          continue; // not in the quant_output_idx list
        }
      }
      if (!do_quant) {
        continue;
      }
      if (isa<tpu::CastOp>(opd.getDefiningOp()) &&
          isF16toF32(cast<tpu::CastOp>(opd.getDefiningOp()))) {
        // if f16 to f32 cast, set output of cast to int8 directly
        auto castOp = cast<tpu::CastOp>(opd.getDefiningOp());
        auto cast_loc = module::getLocLike(castOp.getInput(), "int8");
        double scale;
        int64_t zeropoint = 0;
        module::getScaleAndZeroPoint(castOp.getInput(), scale, zeropoint,
                                     module::isAsymmetric());
        int64_t qmin = -128, qmax = 127;
        uint32_t flag = quant::QuantizationFlags::Signed;
        auto min = module::getCalibratedType(castOp.getInput()).getMin();
        if (min >= 0) {
          qmin = 0;
          qmax = 255;
          flag = 0; // unsigned
        }
        auto qtype = quant::UniformQuantizedType::get(
            flag, IntegerType::get(rewriter.getContext(), 8),
            module::getStorageType(castOp.getInput()), scale, zeropoint, qmin,
            qmax);
        auto cast_type =
            RankedTensorType::get(module::getShape(castOp.getResult()), qtype);
        auto cast_op = rewriter.create<tpu::CastOp>(
            cast_loc, cast_type, ValueRange{castOp.getInput()});
        rewriter.replaceOp(castOp, cast_op.getResult());
        modified = true;
        continue;
      } else { // should not hit
        auto cast_loc = module::getLocLike(opd, "int8");
        auto cast_shape = module::getShape(opd);
        // get scale and zero point from calibrated type
        double scale;
        int64_t zeropoint = 0;
        module::getScaleAndZeroPoint(opd, scale, zeropoint,
                                     module::isAsymmetric());
        int64_t qmin = -128, qmax = 127;
        uint32_t flag = quant::QuantizationFlags::Signed;
        auto min = module::getCalibratedType(opd).getMin();
        if (min >= 0) {
          qmin = 0;
          qmax = 255;
          flag = 0; // unsigned
        }
        auto qtype = quant::UniformQuantizedType::get(
            flag, IntegerType::get(rewriter.getContext(), 8),
            module::getStorageType(opd), scale, zeropoint, qmin, qmax);
        auto cast_type = RankedTensorType::get(cast_shape, qtype);
        auto cast_op =
            rewriter.create<tpu::CastOp>(cast_loc, cast_type, ValueRange{opd});
        op.setOperand(i, cast_op.getOutput());
        modified = true;
      }
    }
    return modified ? success() : failure();
  };
  bool shouldPrint(ReturnOp op) const override { return false; }

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
    if (quant_output_int8 && quant_output_bf16) {
      llvm_unreachable(
          "do not set quant_output_int8 and quant_output_bf16 together");
    }
    if (quant_input) {
      std::vector<int64_t> quant_input_idx = string2vec(quant_input_list);
      patterns.add<StripInputQuantTpuCastPattern>(ctx, quant_input_idx);
      patterns.add<StripInputQuantCpuCastPattern>(ctx, quant_input_idx);
      if (quant_input_int8) {
        patterns.add<ForceInputQuantInt8Pattern>(ctx, quant_input_idx);
      }
    }

    if (quant_output) {
      std::vector<int64_t> quant_output_idx = string2vec(quant_output_list);
      patterns.add<StripOutputQuantTpuCastPattern>(ctx, quant_output_idx);
      patterns.add<StripOutputQuantCpuCastPattern>(ctx, quant_output_idx);
      if (quant_output_int8) {
        patterns.add<ForceOutputQuantInt8Pattern>(ctx, quant_output_idx);
      }
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
