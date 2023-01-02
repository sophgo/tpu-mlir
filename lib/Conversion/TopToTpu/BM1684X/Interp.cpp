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

void InterpLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::InterpOp op) const {
  auto op_ = op.getOperation();
  if(auto a = tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>()))
  {
    op_->setAttr("mode",
                tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(op_->getAttr("coord_mode").cast<StringAttr>())){
    op_->setAttr("coord_mode",
                tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }

  lowering_common_f32<tpu::InterpOp>(rewriter, op);
}
void InterpLowering::LoweringINT4(PatternRewriter &rewriter, top::InterpOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void InterpLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::InterpOp op, bool asymmetric) const {
  auto op_ = op.getOperation();
  if(auto a = tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>()))
  {
    op_->setAttr("mode",
                tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(op_->getAttr("coord_mode").cast<StringAttr>())){
    op_->setAttr("coord_mode",
                tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }
  lowering_common_f16<tpu::InterpOp>(rewriter, op);
}

void InterpLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::InterpOp op) const {
  auto op_ = op.getOperation();
  if(auto a = tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>()))
  {
    op_->setAttr("mode",
                tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(op_->getAttr("coord_mode").cast<StringAttr>())){
    op_->setAttr("coord_mode",
                tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }
  lowering_common_bf16<tpu::InterpOp>(rewriter, op);
}

void InterpLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::InterpOp op) const {
  auto op_ = op.getOperation();
  if(auto a = tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>()))
  {
    op_->setAttr("mode",
                tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(op_->getAttr("coord_mode").cast<StringAttr>())){
    op_->setAttr("coord_mode",
                tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }
  lowering_common_f16<tpu::InterpOp>(rewriter, op);
}

void InterpLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::InterpOp op) const {
  auto op_ = op.getOperation();
  if(auto a = tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>()))
  {
    op_->setAttr("mode",
                tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(op_->getAttr("coord_mode").cast<StringAttr>())){
    op_->setAttr("coord_mode",
                tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }
  lowering_common<tpu::InterpOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
