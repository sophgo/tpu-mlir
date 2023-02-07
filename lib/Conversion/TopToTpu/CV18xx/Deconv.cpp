//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-conv"

namespace tpu_mlir {
namespace cv18xx {
void DeconvLowering::LoweringINT8(PatternRewriter &rewriter, top::DeconvOp op,
                                  bool asymmetric) const {
  auto attr = op.parseParam();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  double in_thr, out_thr;
  in_thr = module::getThreshold(op.getInput());
  out_thr = module::getThreshold(op.getOutput());
  // filter
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  // bias
  i32_array_t bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  if (attr.with_bias) {
    auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  }
  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  int64_t multiplier, rshift;
  int inner_dim = filter_f32->size() / attr.oc;
  // per-channel
  for (int c = 0; c < attr.oc; c++) { // per-channel quantize
    float *p_filter = filter_f32->data() + c * inner_dim;
    float w_max = findMaxabs(p_filter, inner_dim);
    double qscale = getQscaleForFilter(w_max, out_thr, in_thr);
    if (qscale >= 1) {
      // Now cv18xx not support lshift, if qscale > 1, rshift <= 0 not working
      // now we fix threshold_w to limit value qscale = (thr_w * thr_x) / (127.0
      // * thr_y) thr_w = qscale * 127.0 * thr_y / thr_x qscale = 0.99999999
      qscale = 0.999999;
      LLVM_DEBUG(llvm::errs()
                     << "WARNING: adjust threshold_w for qscale"
                     << ", qscale_filter = " << qscale << ", max_filter[" << c
                     << "] = " << qscale * 127 * out_thr / in_thr << "\n";);
    }
    if (attr.with_bias) {
      float b_max = fabs(bias_fp32->data()[c]);
      double qscale_bias = getQscaleForBias(b_max, out_thr);
      if (qscale_bias > qscale) {
        LLVM_DEBUG(llvm::errs() << "WARNING: adjust qscale for bias"
                                << ", qscale_filter = " << qscale
                                << ", qscale_bias = " << qscale_bias << "\n";);
        if (qscale_bias >= 1) {
          // prevent for auto tuning
          LLVM_DEBUG(llvm::errs()
                         << "WARNING:  qscale_bias are valid, keep org qscale"
                         << ", qscale_filter = " << qscale
                         << ", qscale_bias = " << qscale_bias << "\n";);
        } else {
          qscale = qscale_bias;
        }
      }
    }
    // decompose qscale into rshift and multiplier
    getRShiftAndMultiplierFromQScale(qscale, &multiplier, &rshift, true);
    multiplier_v.push_back(multiplier);
    rshift_v.push_back(rshift);
    // quantize weight
    quantizeFilterRShiftAndMultiplier(
        p_filter, filter_i8->data() + c * inner_dim, inner_dim, out_thr, in_thr,
        rshift, multiplier, true);
    if (attr.with_bias) {
      quantizeBiasRShiftAndMultiplier(bias_fp32->data() + c,
                                      bias_int32->data() + c, 1, out_thr,
                                      rshift, multiplier, true);
    }
  }
  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        rewriter.getIntegerType(8, true));
  auto new_filter =
      top::WeightOp::create(op, "filter_i8", *filter_i8, new_type);
  operands.push_back(new_filter);

  if (attr.with_bias) {
    auto new_type =
        RankedTensorType::get({1, attr.oc, 1, 1}, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(op.getBias()); // none
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(ctx, tpu::RequantMode::QDM)));
  attrs.push_back(rewriter.getNamedAttr(
      "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(attr.with_bias)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  auto newOp = rewriter.create<tpu::DeconvOp>(op->getLoc(), newType,
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  Value newValue = newOp.getOutput();
  rewriter.replaceOp(op, {newValue});
}

void DeconvLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::DeconvOp op) const {
  std::vector<Value> operands;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  operands.push_back(op.getInput());
  operands.push_back(filterOp.clone_bf16(op));
  operands.push_back(op.getBias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::DeconvOp>(op, newType, operands, attrs);
}

} // namespace cv18xx
} // namespace tpu_mlir
