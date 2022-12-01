//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {
template <typename T>
static void LoweringInterp(PatternRewriter &rewriter, top::InterpOp op,
                           bool asymmetric) {
  auto mode = tpu::symbolizeResizeMode(op.mode());
  auto coord_mode = tpu::symbolizeResizeCoordMode(op.coord_mode());
  assert(mode && coord_mode);
  std::string coordinate_transformation_mode;
  auto o_shape = Module::getShape(op.output());
  assert(o_shape.size() >= 2);
  switch (coord_mode.value()) {
  case tpu::ResizeCoordMode::half_pixel:
    if (mode.value() == tpu::ResizeMode::nearest) {
      coordinate_transformation_mode = "nearest_half_pixel";
    } else {
      coordinate_transformation_mode = "half_pixel";
    }
    break;
  case tpu::ResizeCoordMode::align_corners:
    coordinate_transformation_mode = "align_corners";
    break;
  case tpu::ResizeCoordMode::pytorch_half_pixel:
    if (mode.value() == tpu::ResizeMode::linear &&
        o_shape[o_shape.size() - 1] > 1 && o_shape[o_shape.size() - 2] > 1) {
      coordinate_transformation_mode = "half_pixel";
    } else {
      coordinate_transformation_mode = "pytorch_half_pixel";
    }
    break;
  default:
    llvm_unreachable("Unsupport interp coord type \n");
  }

  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(rewriter.getNamedAttr("operation_name",
                                           rewriter.getStringAttr("interp")));
  param.emplace_back(rewriter.getNamedAttr(
      "width", rewriter.getI32IntegerAttr(o_shape[o_shape.size() - 1])));
  param.emplace_back(rewriter.getNamedAttr(
      "height", rewriter.getI32IntegerAttr(o_shape[o_shape.size() - 2])));
  param.emplace_back(
      rewriter.getNamedAttr("pad_beg", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(
      rewriter.getNamedAttr("pad_end", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(
      rewriter.getNamedAttr("shrink_factor", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(
      rewriter.getNamedAttr("zoom_factor", rewriter.getI32IntegerAttr(0)));
  param.emplace_back(rewriter.getNamedAttr(
      "coordinate_transformation_mode",
      rewriter.getStringAttr(coordinate_transformation_mode)));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(param)));
  std::vector<Value> operands;
  operands.emplace_back(op.input());
  mlir::Type new_type;
  if (mode.value() == tpu::ResizeMode::nearest) {
    if constexpr (std::is_same_v<T, mlir::IntegerType>) {
      new_type = Quant::getQuantInt8Type(op.output(), asymmetric);
    } else {
      new_type = getQuantBF16Type(op.output());
    }
  } else {
    new_type = getQuantFloatType(op.output());
  }
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_type, operands, attrs);
}
void InterpLowering::LoweringINT8(PatternRewriter &rewriter, top::InterpOp op,
                                  bool asymmetric) const {
  LoweringInterp<mlir::IntegerType>(rewriter, op, asymmetric);
}

void InterpLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::InterpOp op) const {
  LoweringInterp<mlir::BFloat16Type>(rewriter, op, false);
}
} // namespace cv18xx
} // namespace tpu_mlir
