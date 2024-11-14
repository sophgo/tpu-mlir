//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-Csc"
namespace tpu_mlir {
namespace cv18xx {
static inline int align_up(int x, int n) {
  if (n == 0 || n == 1) {
    return x;
  }
  return ((x + n - 1) / n) * n;
}

static void resetCaliType(PatternRewriter &rewriter, top::CscOp op) {
  auto shape = module::getShape(op.getResult());
  quant::CalibratedQuantizedType qtype =
      quant::CalibratedQuantizedType::get(rewriter.getF32Type(), 0, 255);
  RankedTensorType newType = RankedTensorType::get(shape, qtype);
  op.getResult().setType(newType);
}

void CscLowering::LoweringINT8(PatternRewriter &rewriter, top::CscOp op,
                               bool asymmetric) const {
  // lowering_common_int8<tpu::CscOp>(rewriter, op, asymmetric);
  //  auto stype = module::getStorageType(op.getOutput());
  //  assert (stype.isInteger(8) || stype.isUnsignedInteger(8));
  std::vector<Value> operands;
  Value input_val = op.getInput();
  if (module::isUniformQuantized(input_val)) {
    // fuse preprocess has done before.all_int_process() changes the threshold,
    // should reset.
    resetCaliType(rewriter, op);
  }
  operands.emplace_back(input_val);
  auto name = module::getName(op.getOutput());

  std::string pixel_format = op.getPixelFormat().str();
  std::vector<NamedAttribute> attrs;
  int yuv_type = -1;
  bool need_stride_copy = false;
  if (pixel_format == "YUV420_PLANAR") {
    yuv_type = 1;
  } else if (pixel_format == "YUV_NV12") {
    yuv_type = 2;
  } else if (pixel_format == "YUV_NV21") {
    yuv_type = 3;
  } else if (pixel_format == "RGB_PLANAR" || pixel_format == "BGR_PLANAR" ||
             pixel_format == "RGBA_PLANAR") {
    need_stride_copy = true;
  }

  if (yuv_type > 0) {
    attrs.emplace_back(rewriter.getNamedAttr("y_align", op.getYAlignAttr()));
    attrs.emplace_back(rewriter.getNamedAttr("w_align", op.getWAlignAttr()));
    attrs.emplace_back(
        rewriter.getNamedAttr("channel_align", op.getChannelAlignAttr()));
    attrs.emplace_back(
        rewriter.getNamedAttr("pixel_format", op.getPixelFormatAttr()));
    attrs.emplace_back(rewriter.getNamedAttr(
        "pixel_type", rewriter.getI64IntegerAttr(yuv_type)));
    auto newType = getQuantInt8Type(op.getOutput());
    rewriter.replaceOpWithNewOp<tpu::CscOp>(op, newType, operands, attrs);
  } else if (need_stride_copy) {
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t n, c, h, w;
    int64_t on, oc, oh, ow;
    module::getNCHW(op.getInput(), n, c, h, w, false);
    module::getNCHW(op.getOutput(), on, oc, oh, ow, false);
    std::vector<int64_t> i_stride(4, 0);
    std::vector<int64_t> o_stride(4, 0);
    std::vector<int64_t> copy_shape(4, 1);
    i_stride[3] = 1;
    i_stride[2] = align_up(ow, op.getWAlign());
    i_stride[1] = align_up(i_stride[2] * oh, op.getChannelAlign());
    i_stride[0] = i_stride[1] * c;

    o_stride[3] = 1;
    o_stride[2] = ow;
    o_stride[1] = o_stride[2] * oh;
    o_stride[0] = i_stride[1] * oc;
    copy_shape[3] = ow;
    copy_shape[2] = oh;
    copy_shape[1] = oc;
    copy_shape[0] = on;
    attrs.emplace_back(
        rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(copy_shape)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "input_stride", rewriter.getI64ArrayAttr(i_stride)));
    attrs.emplace_back(rewriter.getNamedAttr(
        "output_stride", rewriter.getI64ArrayAttr(o_stride)));
    auto caliType = module::getCalibratedType(op.getOutput());
    auto newType = RankedTensorType::get({on, oc, oh, ow}, caliType);
    rewriter.replaceOpWithNewOp<top::CopyOp>(op, newType, operands, attrs);
  } else {
    rewriter.setInsertionPointAfterValue(input_val);
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    int64_t n, c, h, w;
    int64_t on, oc, oh, ow;
    module::getNCHW(op.getInput(), n, c, h, w, false);
    module::getNCHW(op.getOutput(), on, oc, oh, ow, false);
    int64_t unaligned_w = oc * oh * ow / (c * h);
    std::vector<int64_t> slice_offset{0, 0, 0, 0};
    std::vector<int64_t> slice_stpes{1, 1, 1, 1};
    std::vector<int64_t> slice_ends{-1, -1, -1, -1};
    auto caliType = module::getCalibratedType(op.getOutput());
    auto slice_type = RankedTensorType::get({n, c, h, unaligned_w}, caliType);
    attrs.emplace_back(rewriter.getNamedAttr(
        "offset", rewriter.getI64ArrayAttr(slice_offset)));
    attrs.emplace_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(slice_stpes)));
    attrs.emplace_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(slice_ends)));
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_slice"));
    auto noneOp = module::getNoneOp(op);
    for (int i = 0; i < 3; i++) {
      operands.emplace_back(noneOp);
    }
    auto slice_op =
        rewriter.create<top::SliceOp>(loc, slice_type, operands, attrs);
    auto slice_out = slice_op.getOutput();
    attrs.clear();
    std::vector<Value> reshape_operands;
    reshape_operands.emplace_back(slice_out);
    auto reshape_type = RankedTensorType::get({on, oc, oh, ow}, caliType);
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(op, reshape_type,
                                                reshape_operands, attrs);
  }
}

void CscLowering::LoweringBF16(PatternRewriter &rewriter, top::CscOp op) const {
  LoweringINT8(rewriter, op, false);
}
} // namespace cv18xx
} // namespace tpu_mlir
