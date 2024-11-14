//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void InterpLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::InterpOp op) const {
  auto op_ = op.getOperation();
  if (auto a =
          tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>())) {
    op_->setAttr("mode",
                 tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(
          op_->getAttr("coord_mode").cast<StringAttr>())) {
    op_->setAttr("coord_mode",
                 tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }
  lowering_common_f32<tpu::InterpOp>(rewriter, op, 3);
}

void InterpLowering::LoweringINT8(PatternRewriter &rewriter, top::InterpOp op,
                                  bool asymmetric) const {
  auto op_ = op.getOperation();
  if (auto a =
          tpu::symbolizeResizeMode(op_->getAttr("mode").cast<StringAttr>())) {
    op_->setAttr("mode",
                 tpu::ResizeModeAttr::get(op_->getContext(), a.value()));
  }

  if (auto a = tpu::symbolizeResizeCoordMode(
          op_->getAttr("coord_mode").cast<StringAttr>())) {
    op_->setAttr("coord_mode",
                 tpu::ResizeCoordModeAttr::get(op_->getContext(), a.value()));
  }
  // seems like NNTC's interp only support F32
  // bmcompiler/src/interface/bmcompiler_if.cpp :813
  lowering_common_f32<tpu::InterpOp>(rewriter, op, 3);
}

} // namespace bm1684
} // namespace tpu_mlir
