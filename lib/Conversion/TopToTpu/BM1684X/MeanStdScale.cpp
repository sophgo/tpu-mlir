//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

#define DEBUG_TYPE "lowering-MeanStdScale"
namespace tpu_mlir {
namespace bm1684x {

static void QuantizeMultiplier(float value, int length, int bits, int *multiplier, int *shift) {
  for (int i = 0; i < length; ++i) {
    if (value != 0.0) {
      shift[i] = -(floor(log2(fabs(value))) + 1);
      shift[i] += bits - 1;
      multiplier[i] = round(value * pow(2, shift[i]));

      if (multiplier[i] >= (1 << (bits - 1)) ||
          multiplier[i] < -(1 << (bits - 1))) {
        shift[i] -= 1;
        multiplier[i] = multiplier[i] >> 1;
      }
    } else {
      multiplier[i] = 0;
      shift[i] = 0;
    }
  }
}

static void op_lowering_common(PatternRewriter &rewriter,
                                      top::MeanStdScaleOp op) {
  auto processOp = op.getOperation();
  assert(processOp);

  MLIRContext *context = processOp->getContext();

  //multiplier & rshift
  std::vector<Attribute> vector_multi;
  std::vector<Attribute> vector_rshift;
  std::vector<Attribute> vector_offset;

  auto idtype = module::getStorageType(processOp->getOperands()[0].getType());
  auto odtype = module::getStorageType(processOp->getResult(0).getType());
  auto scale_attr = op.getScale();
  auto std_attr = op.getStd();
  auto mean_attr = op.getMean();
  auto scale = *module::getF64Array(scale_attr);
  auto std = *module::getF64Array(std_attr);
  auto mean = * module::getF64Array(mean_attr);
  int chn_num = std.size();

  if ((idtype.isUnsignedInteger(8) || idtype.isSignedInteger(8)) && odtype.isSignedInteger(8)) {
    int multi = 0;
    int rshift = 0;
    for (int i = 0; i < chn_num; i++) {
      QuantizeMultiplier(scale[0] / (std[i] * scale[1]), 1, 16, &multi, &rshift);
      vector_multi.push_back(IntegerAttr::get(rewriter.getI32Type(), multi));
      vector_rshift.push_back(IntegerAttr::get(rewriter.getI32Type(), rshift));
      int offset = (int)roundf(0 - mean[i] * (scale[0] / (std[i] * scale[1])) *pow(2, rshift));
      vector_offset.push_back(IntegerAttr::get(rewriter.getI32Type(), offset));
    }
  }
  else if (idtype.isF32() && odtype.isSignedInteger(8)) {
    int multi = 0;
    int rshift = 0;
    for (int i = 0; i < chn_num; i++) {
      QuantizeMultiplier(scale[0] / (1 / scale[1]), 1, 16, &multi, &rshift);
      vector_multi.push_back(IntegerAttr::get(rewriter.getI32Type(), multi));
      vector_rshift.push_back(IntegerAttr::get(rewriter.getI32Type(), rshift));
      vector_offset.push_back(IntegerAttr::get(rewriter.getI32Type(), 0));
    }
  } else {
    int multi = 1;
    int rshift = 0;
    for (int i = 0; i < chn_num; i++) {
      vector_multi.push_back(IntegerAttr::get(rewriter.getI32Type(), multi));
      vector_rshift.push_back(IntegerAttr::get(rewriter.getI32Type(), rshift));
      vector_offset.push_back(IntegerAttr::get(rewriter.getI32Type(), 0));
    }
  }

  processOp->setAttr("multi", ArrayAttr::get(context, vector_multi));
  processOp->setAttr("rshift", ArrayAttr::get(context, vector_rshift));
  processOp->setAttr("offset", ArrayAttr::get(context, vector_offset));

  lowering_common<tpu::MeanStdScaleOp>(rewriter, processOp, processOp->getResult(0).getType(), 1);
}

void MeanStdScaleLowering::LoweringINT8(PatternRewriter &rewriter,
                                          top::MeanStdScaleOp op,
                                          bool asymmetric) const {
  lowering_common_int8<tpu::MeanStdScaleOp>(rewriter, op, asymmetric, 1);
}

void MeanStdScaleLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::MeanStdScaleOp op,
                                        bool asymmetric) const {
  lowering_common_int8<tpu::MeanStdScaleOp>(rewriter, op, asymmetric, 1);
}

void MeanStdScaleLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::MeanStdScaleOp op) const {
  lowering_common_bf16<tpu::MeanStdScaleOp>(rewriter, op, 1);
}

void MeanStdScaleLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::MeanStdScaleOp op) const {
  lowering_common_f32<tpu::MeanStdScaleOp>(rewriter, op, 1);
}

void MeanStdScaleLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::MeanStdScaleOp op) const {
  op_lowering_common(rewriter, op);
}

void MeanStdScaleLowering::LoweringF8(PatternRewriter &rewriter,
                                     top::MeanStdScaleOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanStdScaleLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::MeanStdScaleOp op) const {
  op_lowering_common(rewriter, op);
}

}
}
