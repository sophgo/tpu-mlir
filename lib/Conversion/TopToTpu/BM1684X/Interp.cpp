//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {
static void LoweringInterp(PatternRewriter &rewriter, top::InterpOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  assert(nInputs == 2);
  for (auto i = 0; i < nInputs; ++i) {
    auto opd = op->getOperand(i);
    if (module::isWeight(opd)) {
      //remove target_shape operands of top::InterpOp op
      opd.dropAllUses();
      opd.getDefiningOp()->erase();
      auto v = module::getNoneOp(op);
      operands.push_back(v);
    } else {
      operands.push_back(opd);
    }
  }

  if(auto a = tpu::symbolizeResizeMode(op->getAttr("mode").cast<StringAttr>()))
  {
    op->setAttr("mode",
                tpu::ResizeModeAttr::get(op->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(op->getAttr("coord_mode").cast<StringAttr>())){
    op->setAttr("coord_mode",
                tpu::ResizeCoordModeAttr::get(op->getContext(), a.value()));
  }


  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  if (type.isF32()) {
    rewriter.replaceOpWithNewOp<tpu::InterpOp>(op, op->getResultTypes(), operands,
                                             attrs);
    return;
  }
  std::vector<Type> new_types;
  for (auto out : op->getResults()) {
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
  }
  rewriter.replaceOpWithNewOp<tpu::InterpOp>(op, new_types, operands, attrs);
  return;
}

void InterpLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::InterpOp op) const {
  LoweringInterp(rewriter, op, rewriter.getF32Type());
}
void InterpLowering::LoweringINT4(PatternRewriter &rewriter, top::InterpOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void InterpLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::InterpOp op, bool asymmetric) const {
   LoweringInterp(rewriter, op, rewriter.getF16Type());
}

void InterpLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::InterpOp op) const {
  LoweringInterp(rewriter, op, rewriter.getBF16Type());
}

void InterpLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::InterpOp op) const {
  LoweringInterp(rewriter, op, rewriter.getF16Type());
}

void InterpLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::InterpOp op) const {
  LoweringInterp(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
