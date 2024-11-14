//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include <string.h>
#define DEBUG_TYPE "lowering-MeanStdScale"
namespace tpu_mlir {
namespace bm1684x {

static void QuantizeMultiplier(float value, int length, int bits,
                               int *multiplier, int *shift) {
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
                               top::MeanStdScaleOp op, Type type) {
  auto processOp = op.getOperation();
  assert(processOp);

  MLIRContext *context = processOp->getContext();

  // multiplier & rshift
  std::vector<Attribute> vector_multi;
  std::vector<Attribute> vector_rshift;
  std::vector<Attribute> vector_offset;
  std::vector<float> f32Param;

  auto idtype = module::getStorageType(processOp->getOperands()[0].getType());
  auto odtype = module::getStorageType(type);
  assert(odtype.isSignedInteger(8) || odtype.isF16());
  auto scale_attr = op.getScale();
  auto std_attr = op.getStd();
  auto mean_attr = op.getMean();
  auto scale = *module::getF64Array(scale_attr);
  auto std = *module::getF64Array(std_attr);
  auto mean = *module::getF64Array(mean_attr);
  int chn_num = std.size();

  if ((idtype.isUnsignedInteger(8) || idtype.isSignedInteger(8)) &&
      odtype.isSignedInteger(8)) {
    int multi = 0;
    int rshift = 0;
    int sub_rshift = 0;
    for (int i = 0; i < chn_num; i++) {
      QuantizeMultiplier(scale[0] / (std[i] * scale[1]), 1, 16, &multi,
                         &rshift);
      sub_rshift = -rshift;
      vector_multi.push_back(IntegerAttr::get(rewriter.getI32Type(), multi));
      vector_rshift.push_back(IntegerAttr::get(rewriter.getI32Type(), rshift));
      int offset = (int)roundf(0 - mean[i] * (scale[0] / (std[i] * scale[1])) *
                                       pow(2, rshift));
      vector_offset.push_back(IntegerAttr::get(rewriter.getI32Type(), offset));
      f32Param.push_back(double(1.0) / std[i]);
      f32Param.push_back(mean[i]);
      float floatValue1, floatValue2, floatValue3;
      std::memcpy(&floatValue1, &multi, sizeof(int32_t));
      std::memcpy(&floatValue2, &sub_rshift, sizeof(int32_t));
      std::memcpy(&floatValue3, &offset, sizeof(int32_t));
      f32Param.push_back(floatValue1);
      f32Param.push_back(floatValue2);
      f32Param.push_back(floatValue3);
    }
  } else if (idtype.isF32() && odtype.isSignedInteger(8)) {
    int multi = 0;
    int rshift = 0;
    int sub_rshift = 0;
    for (int i = 0; i < chn_num; i++) {
      QuantizeMultiplier(scale[0] / (1 / scale[1]), 1, 16, &multi, &rshift);
      sub_rshift = -rshift;
      vector_multi.push_back(IntegerAttr::get(rewriter.getI32Type(), multi));
      vector_rshift.push_back(IntegerAttr::get(rewriter.getI32Type(), rshift));
      vector_offset.push_back(IntegerAttr::get(rewriter.getI32Type(), 0));
      f32Param.push_back(double(1.0) / std[i]);
      f32Param.push_back(mean[i]);
      float floatValue1, floatValue2;
      std::memcpy(&floatValue1, &multi, sizeof(int32_t));
      std::memcpy(&floatValue2, &sub_rshift, sizeof(int32_t));
      f32Param.push_back(floatValue1);
      f32Param.push_back(floatValue2);
      f32Param.push_back(0);
    }
  } else {
    int multi = 1;
    int rshift = 0;
    int sub_rshift = 0;
    for (int i = 0; i < chn_num; i++) {
      vector_multi.push_back(IntegerAttr::get(rewriter.getI32Type(), multi));
      vector_rshift.push_back(IntegerAttr::get(rewriter.getI32Type(), rshift));
      vector_offset.push_back(IntegerAttr::get(rewriter.getI32Type(), 0));
      f32Param.push_back(double(1.0) / std[i]);
      f32Param.push_back(mean[i]);
      float floatValue1, floatValue2;
      std::memcpy(&floatValue1, &multi, sizeof(int32_t));
      std::memcpy(&floatValue2, &sub_rshift, sizeof(int32_t));
      f32Param.push_back(floatValue1);
      f32Param.push_back(floatValue2);
      f32Param.push_back(0);
    }
  }

  auto f32ParamType =
      RankedTensorType::get({1, chn_num, 1, 5}, rewriter.getF32Type());
  auto f32ParamTensor =
      top::WeightOp::create(op, "f32param", f32Param, f32ParamType);
  auto original_name = module::getName(op.getOperation()).str();

  auto roundmodeflag = 0;
  if ((idtype.isF32() || idtype.isUnsignedInteger(8) ||
       idtype.isSignedInteger(8)) &&
      odtype.isF16()) {
    roundmodeflag = 1;
  }

  auto newOp = rewriter.create<tpu::MeanStdScaleOp>(
      op.getLoc(), type, op.getInput(), f32ParamTensor, op.getQuantMode(),
      op.getCustomizationFormat(), op.getChannelOrder(), op.getSign(),
      scale_attr, std_attr, mean_attr, op.getZeroPoints(), op.getResizeDims(),
      ArrayAttr::get(context, vector_multi),
      ArrayAttr::get(context, vector_rshift),
      ArrayAttr::get(context, vector_offset),
      roundmodeflag == 0
          ? op.getRoundingMode()
          : StringAttr::get(op.getContext(), "HalfToEven").getValue());

  rewriter.replaceOp(op, newOp.getOutput());
}

void MeanStdScaleLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::MeanStdScaleOp op,
                                        bool asymmetric) const {
  // lowering_common_int8<tpu::MeanStdScaleOp>(rewriter, op, asymmetric, 1);
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanStdScaleLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::MeanStdScaleOp op,
                                        bool asymmetric) const {
  // lowering_common_int8<tpu::MeanStdScaleOp>(rewriter, op, asymmetric, 1);
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanStdScaleLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::MeanStdScaleOp op) const {
  // lowering_common_bf16<tpu::MeanStdScaleOp>(rewriter, op, 1);
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanStdScaleLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::MeanStdScaleOp op) const {
  // lowering_common_f32<tpu::MeanStdScaleOp>(rewriter, op, 1);
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanStdScaleLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::MeanStdScaleOp op) const {
  auto new_type = getQuantFloatType<mlir::Float16Type>(op.getOutput());
  op_lowering_common(rewriter, op, new_type);
}

void MeanStdScaleLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::MeanStdScaleOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void MeanStdScaleLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::MeanStdScaleOp op) const {
  auto new_type = op.getOutput().getType();
  op_lowering_common(rewriter, op, new_type);
}

} // namespace bm1684x
} // namespace tpu_mlir
