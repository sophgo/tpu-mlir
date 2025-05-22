//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "Common.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"

using namespace llvm;
using namespace tpu_mlir::backend;
namespace tpu_mlir {

namespace bm1684x {

/*
One unbalanceCast Connects to Add.
Viewed as a specifc case for more general linear-ratio throughout case

CastOp[GlobalorNULL] --->
                         AddOp     =>  UnbalanceAddOp2
CastOp[GlobalorNULL] --->

only support Cast: i8 => bf16
*/
class FuseCastAddPattern : public OpRewriterPatternEx<tpu::AddOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseCastAddPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::AddOp>(context, "FuseCastAddPattern",
                                        benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::AddOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto leftOp =
        dyn_cast_or_null<tpu::CastOp>(op.getOperand(0).getDefiningOp());
    auto rightOp =
        dyn_cast_or_null<tpu::CastOp>(op.getOperand(1).getDefiningOp());
    int leftValid = 0, rightValid = 0;
    auto op_name = module::getName(op.getResult()).str();
    op_name = op_name;

    if (leftOp) {
      if (!leftOp->hasOneUse())
        return failure();
      auto leftOutputType = module::getElementType(leftOp.getResult());
      if ((!module::isUniformQuantized(leftOp.getInput())) ||
          (!leftOutputType.isBF16())) {
        leftValid = 0;
      } else {
        leftValid = 1;
      }
    }

    if (rightOp) {
      if (!rightOp->hasOneUse())
        return failure();
      auto rightOutputType = module::getElementType(rightOp.getResult());
      if ((!module::isUniformQuantized(rightOp.getInput())) ||
          (!rightOutputType.isBF16())) {
        rightValid = 0;
      } else {
        rightValid = 1;
      }
    }
    if (!(leftValid || rightValid))
      return failure();

    std::vector<NamedAttribute> attrs;
    rewriter.setInsertionPointAfter(op);
    auto type_out = op.getResult().getType();
    auto new_castAdd_op = rewriter.create<tpu::CastAddOp>(
        NameLoc::get(rewriter.getStringAttr(op_name)), //+ "_CastAdd"
        type_out, op->getOperands(), attrs);
    new_castAdd_op->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
    new_castAdd_op->setAttr(
        "relu_limit",
        rewriter.getF64FloatAttr(op.getReluLimit().convertToDouble()));

    auto output = op.getOutput();
    output.replaceAllUsesExcept(new_castAdd_op.getOutput(), new_castAdd_op);
    op.erase();
    if (leftValid) {
      new_castAdd_op.setOperand(0, leftOp.getInput());
      new_castAdd_op.setRoundModeAttr(leftOp.getRoundModeAttr());
      leftOp.erase();
    }
    if (rightValid) {
      new_castAdd_op.setOperand(1, rightOp.getInput());
      new_castAdd_op.setRoundModeAttr(rightOp.getRoundModeAttr());
      rightOp.erase();
    }
    return success();
  }
};

/*
LayerNorm -> Reshape -> Cast => LayerNorm -> Cast -> Reshape
*/
class MoveReshapePattern : public OpRewriterPatternEx<tpu::ReshapeOp> {
public:
  MoveReshapePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ReshapeOp>(context, "MoveReshapePattern",
                                            benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    if (!op->hasOneUse()) {
      return failure();
    }
    auto op_name = module::getName(op.getOutput()).str();
    op_name = op_name;
    auto output = op.getOutput();
    Operation *castOp = *output.user_begin();
    if (auto layerNormOp =
            dyn_cast_or_null<tpu::LayerNormOp>(op.getInput().getDefiningOp())) {
      if (!layerNormOp->hasOneUse()) {
        return failure();
      }
      auto ishape = module::getShape(op.getInput());
      if (auto castOpInst_2 = dyn_cast_or_null<tpu::CastOp>(castOp)) {
        op.replaceAllUsesWith(op.getInput());
        auto next_out = castOpInst_2.getResult();
        auto ori_loc = castOpInst_2.getLoc();
        module::setLocSuffix(castOpInst_2, "reshape_down");
        auto castOpInst_out = castOpInst_2.getResult();
        module::setShape(castOpInst_out, ishape);
        module::setShape(next_out, ishape);
        rewriter.setInsertionPointAfterValue(next_out);
        auto out_shape = module::getShape(op.getOutput()).vec();
        auto reshape_type = module::getTypeLike(next_out, out_shape);
        auto shapeAttr = op.getShape();
        auto new_reshape_op = rewriter.create<tpu::ReshapeOp>(
            ori_loc, reshape_type, ValueRange{next_out},
            rewriter.getNamedAttr("shape", shapeAttr));
        rewriter.replaceAllUsesExcept(next_out, new_reshape_op.getOutput(),
                                      new_reshape_op);
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

/*
Cast(Dequant) + LayerNorm => LayerNormCast.cpp
only support Cast: i8 => bf16
*/
class FuseDequantLayerNormPattern
    : public OpRewriterPatternEx<tpu::LayerNormOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseDequantLayerNormPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::LayerNormOp>(
            context, "FuseDequantLayerNormPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::LayerNormOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto op_name = module::getName(op.getResult()).str();
    op_name = op_name;

    auto FrontOp =
        dyn_cast_or_null<tpu::CastOp>(op.getOperand(0).getDefiningOp());
    int castValid = 0;

    if (FrontOp) {
      if (!FrontOp->hasOneUse())
        return failure();
      auto leftOutputType = module::getElementType(FrontOp.getResult());
      if ((!module::isUniformQuantized(FrontOp.getInput())) ||
          (!leftOutputType.isBF16())) {
        castValid = 0;
      } else {
        castValid = 1;
      }
    }

    if (!(castValid))
      return failure();

    std::vector<NamedAttribute> attrs;
    rewriter.setInsertionPointAfter(op);
    auto type_out = op.getResult().getType();
    std::vector<Value> opds;
    opds.reserve(3);
    const int nInputs = 3;
    assert(nInputs <= op->getNumOperands());
    for (auto i = 0; i < nInputs; ++i) {
      auto opd = op->getOperand(i);
      opds.push_back(opd);
    }
    auto new_layer_norm_op = rewriter.create<tpu::LayerNormCastOp>(
        NameLoc::get(rewriter.getStringAttr(op_name)), //+ "_CastAdd"
        type_out, opds, attrs);

    auto output = op.getOutput();
    output.replaceAllUsesExcept(new_layer_norm_op.getOutput(),
                                new_layer_norm_op);
    op.erase();
    if (castValid) {
      new_layer_norm_op.setOperand(0, FrontOp.getInput());
      new_layer_norm_op.setRoundModeAttr(FrontOp.getRoundModeAttr());
      new_layer_norm_op.setAxisAttr(op.getAxisAttr());
      new_layer_norm_op.setEpsAttr(op.getEpsAttr());
      new_layer_norm_op.setIsCastAtEnd(0); // Cast+LN is 0-flag
      FrontOp.erase();
    }
    return success();
  }
};

/*
LayerNorm + Cast(Requant) => LayerNormCast.cpp

only support Cast: bf16 => i8
*/
class FuseLayerNormCastPattern : public OpRewriterPatternEx<tpu::LayerNormOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseLayerNormCastPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::LayerNormOp>(
            context, "FuseLayerNormCastPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::LayerNormOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto op_name = module::getName(op.getResult()).str();
    op_name = op_name;

    bool castValid = 0;
    auto next_op =
        dyn_cast_or_null<tpu::CastOp>(*op.getResult().getUsers().begin());
    if (next_op) {
      // if(!next_op->hasOneUse()) return failure();
      auto CastInputType = module::getStorageType(next_op.getInput());
      if ((!module::isUniformQuantized(next_op.getOutput())) ||
          (!CastInputType.isBF16())) {
        castValid = 0;
      } else {
        castValid = 1;
      }
    }

    if (!(castValid))
      return failure();

    std::vector<NamedAttribute> attrs;
    rewriter.setInsertionPointAfter(op);
    std::vector<Value> opds;
    opds.reserve(3);
    const int nInputs = 3;
    assert(nInputs <= op->getNumOperands());
    for (auto i = 0; i < nInputs; ++i) {
      auto opd = op->getOperand(i);
      opds.push_back(opd);
    }

    auto type_out = next_op.getResult().getType();
    auto cast_op_name = module::getName(next_op.getResult()).str();
    cast_op_name = cast_op_name;
    auto new_layer_norm_op = rewriter.create<tpu::LayerNormCastOp>(
        NameLoc::get(rewriter.getStringAttr(op_name)), //+ "_CastAdd"
        type_out, opds, attrs);

    auto Castoutput = next_op.getOutput();
    Castoutput.replaceAllUsesExcept(new_layer_norm_op.getOutput(),
                                    new_layer_norm_op);
    if (castValid) {
      new_layer_norm_op.setRoundModeAttr(next_op.getRoundModeAttr());
      new_layer_norm_op.setAxisAttr(op.getAxisAttr());
      new_layer_norm_op.setEpsAttr(op.getEpsAttr());
      new_layer_norm_op.setIsCastAtEnd(1); // Cast+LN is 1-flag
      next_op.erase();
    }
    return success();
  }
};

/*
LuT + MatMul
However,  Matmul possibly reuse inputs, carefully apply such pattern.
*/
class FuseLutMatMulPattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseLutMatMulPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "FuseLutMatMulPattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto LutOp = dyn_cast_or_null<tpu::LutOp>(op.getOperand(0).getDefiningOp());
    int LutValid = 0;
    auto op_name = module::getName(op.getResult()).str();
    op_name = op_name;

    if (LutOp) {
      if (!LutOp->hasOneUse())
        return failure();
      LutValid = 1;
    }
    if (!(LutValid))
      return failure();
    std::vector<Value> opds;
    opds.reserve(6); // Input,table,right,bias,mulit,buffer
    for (auto i = 0; i < 2; ++i) {
      auto opd = LutOp->getOperand(i);
      opds.push_back(opd);
    }
    for (auto i = 1; i < 5; ++i) {
      auto opd = op->getOperand(i);
      opds.push_back(opd);
    }
    rewriter.setInsertionPointAfter(op);
    auto type_out = op.getResult().getType();
    auto new_lutMatMul_op = rewriter.create<tpu::MatMulLutOp>(
        NameLoc::get(rewriter.getStringAttr(op_name)), //+ "_CastAdd"
        type_out, opds, op->getAttrs());
    auto output = op.getOutput();
    output.replaceAllUsesExcept(new_lutMatMul_op.getOutput(), new_lutMatMul_op);
    op.erase();
    return success();
  }
};

/*
MatMul + Lut-> MatMul+Lut
*/
class FuseMatMulLutPattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseMatMulLutPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "FuseMatMulLutPattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto op_name = module::getName(op.getResult()).str();
    op_name = op_name;
    auto LutOp =
        dyn_cast_or_null<tpu::LutOp>(*op.getResult().getUsers().begin());
    int LutValid = 0;
    if (LutOp) {
      LutValid = 1;
    }
    if (!(LutValid))
      return failure();
    std::vector<Value> opds;
    opds.reserve(6); // Input,table,right,bias,mulit,buffer
    opds.push_back(op->getOperand(0));
    opds.push_back(LutOp->getOperand(1));
    for (auto i = 1; i < 5; ++i) {
      auto opd = op->getOperand(i);
      opds.push_back(opd);
    }
    rewriter.setInsertionPointAfter(LutOp);
    auto type_out = LutOp.getResult().getType();
    auto new_lutMatMul_op = rewriter.create<tpu::MatMulLutOp>(
        NameLoc::get(rewriter.getStringAttr(op_name)), //+ "_CastAdd"
        type_out, opds, op->getAttrs());
    auto outputLut = LutOp.getOutput();
    outputLut.replaceAllUsesExcept(new_lutMatMul_op.getOutput(),
                                   new_lutMatMul_op);
    LutOp.erase();
    return success();
  }
};

class ActiveCastFusePattern : public OpRewriterPatternEx<tpu::CastOp> {
public:
  ActiveCastFusePattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::CastOp>(context, "ActiveCastFusePattern",
                                         benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::CastOp castOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (castOp->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto in = castOp.getInput();
    auto def_op = in.getDefiningOp();
    if (auto activeOp = dyn_cast_or_null<tpu::ActiveOp>(def_op)) {
      if (!activeOp->hasOneUse())
        return failure();
      bool is_expected_mode = false;
      switch (activeOp.getMode()) {
      case tpu::ActiveMode::GELU:
      case tpu::ActiveMode::TGELU:
      case tpu::ActiveMode::QGELU:
        is_expected_mode = true;
      default:;
      }
      if (!is_expected_mode)
        return failure();
      std::vector<NamedAttribute> attrs;
      for (auto &attr : activeOp->getAttrs()) {
        attrs.push_back(attr);
      }
      for (auto &attr : castOp->getAttrs()) {
        attrs.push_back(attr);
      }
      auto out = castOp.getOutput();
      auto newOp = rewriter.create<tpu::FusedActiveCastOp>(
          out.getLoc(), out.getType(), activeOp.getInput(), attrs);
      out.replaceAllUsesWith(newOp.getOutput());
    } else {
      return failure();
    }
    return success();
  }
};

/*
Softmax + Cast(Requant) => SoftmaxCast.cpp

only support Cast: bf16 => i8
*/
class FuseSoftmaxCastPattern : public OpRewriterPatternEx<tpu::SoftmaxOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FuseSoftmaxCastPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::SoftmaxOp>(context, "FuseSoftmaxCastPattern",
                                            benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::SoftmaxOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isMARS3())
      return failure();
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return failure();
    auto output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto op_name = module::getName(op.getResult()).str();
    op_name = op_name;

    bool castValid = 0;
    auto next_op =
        dyn_cast_or_null<tpu::CastOp>(*op.getResult().getUsers().begin());
    if (next_op) {
      // if(!next_op->hasOneUse()) return failure();
      auto CastInputType = module::getStorageType(next_op.getInput());
      if ((!module::isUniformQuantized(next_op.getOutput())) ||
          (!CastInputType.isBF16())) {
        castValid = 0;
      } else {
        castValid = 1;
      }
    }

    if (!(castValid))
      return failure();

    std::vector<NamedAttribute> attrs;
    rewriter.setInsertionPointAfter(op);
    std::vector<Value> opds;
    opds.reserve(6);
    const int nInputs = 6;
    assert(nInputs <= op->getNumOperands());
    for (auto i = 0; i < nInputs; ++i) {
      auto opd = op->getOperand(i);
      opds.push_back(opd);
    }

    auto type_out = next_op.getResult().getType();
    auto cast_op_name = module::getName(next_op.getResult()).str();
    cast_op_name = cast_op_name;
    auto new_softmax_op = rewriter.create<tpu::SoftmaxCastOp>(
        NameLoc::get(rewriter.getStringAttr(op_name)), type_out, opds, attrs);

    auto Castoutput = next_op.getOutput();
    Castoutput.replaceAllUsesExcept(new_softmax_op.getOutput(), new_softmax_op);
    if (castValid) {
      new_softmax_op.setAxisAttr(op.getAxisAttr());
      new_softmax_op.setRoundModeAttr(next_op.getRoundModeAttr());
      new_softmax_op.setLogAttr(op.getLogAttr());
      new_softmax_op.setBetaAttr(op.getBetaAttr());
      next_op.erase();
    }
    return success();
  }
};
} // namespace bm1684x

namespace tpu {
using namespace bm1684x;
void populateOptPostProcessorBM1684XPatterns(RewritePatternSet *patterns) {
  auto ctx = patterns->getContext();
  patterns->add<FuseCastAddPattern, MoveReshapePattern,
                FuseLayerNormCastPattern, FuseMatMulLutPattern,
                ActiveCastFusePattern, FuseSoftmaxCastPattern>(ctx, 10);
}
} // namespace tpu

} // namespace tpu_mlir
