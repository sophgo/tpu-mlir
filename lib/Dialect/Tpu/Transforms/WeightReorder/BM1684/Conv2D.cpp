//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../WeightReorder.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/WinoGrad.h"

using namespace bm1684;

void conv_weight_transform(int ic, int oc, int kh, int kw,
                           const void *weight_orig, const void *weight_trans,
                           int type_bytes) {
  int trans_offset;
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kh * kw; k_idx++) {
        int orig_offset = ic_idx * kh * kw + k_idx + oc_idx * kh * kw * ic;
        switch (type_bytes) {
        case 4:
          trans_offset = ic_idx + k_idx * align_up(ic, 2) +
                         oc_idx * kh * kw * align_up(ic, 2);
          *((float *)weight_trans + trans_offset) =
              *((float *)weight_orig + orig_offset);
          break;
        case 1:
          trans_offset = ic_idx + k_idx * align_up(ic, 4) +
                         oc_idx * kh * kw * align_up(ic, 4);
          *((char *)weight_trans + trans_offset) =
              *((char *)weight_orig + orig_offset);
          break;
        case 2:
          trans_offset = ic_idx + k_idx * ic + oc_idx * kh * kw * ic;
          *((short *)weight_trans + trans_offset) =
              *((short *)weight_orig + orig_offset);
          break;
        default:
          llvm_unreachable("wrong conv weight data type");
        }
      }
    }
  }
}

LogicalResult convReorder(tpu::Conv2DOp op, PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  auto type_bytes = 1;
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = module::getElementType(op.getFilter());
  std::vector<int64_t> new_shape = {1, attr.oc, attr.kh * attr.kw,
                                    align_up(attr.ic / attr.groups, 4)};
  if ((attr.dh > 15 || attr.dw > 15) && attr.groups == 1) {
    int64_t factor_dh = 2, factor_dw = 2;
    int64_t new_kh = attr.kh, new_kw = attr.kw;
    int64_t new_dh = attr.dh, new_dw = attr.dw;
    if (attr.dh > 15) {
      while (factor_dh <= 15 &&
             (attr.dh % factor_dh || attr.dh / factor_dh > 15))
        factor_dh++;
      if (factor_dh > 15) {
        llvm_unreachable("Un-supported dh by conv layer");
        op.dump();
        return failure();
      }
      new_dh /= factor_dh;
      new_kh = (attr.kh - 1) * factor_dh + 1;
    }
    if (attr.dw > 15) {
      while (factor_dw <= 15 &&
             (attr.dw % factor_dw || attr.dw / factor_dw > 15))
        factor_dw++;
      if (factor_dw > 15) {
        llvm_unreachable("Un-supported dh by conv layer");
        op.dump();
        return failure();
      }
      new_dw /= factor_dw;
      new_kw = (attr.kw - 1) * factor_dw + 1;
    }
    auto input_shape = module::getShape(op->getOperand(0));
    auto filter_op = op.getFilter().getDefiningOp<top::WeightOp>();
    auto filter_type = module::getStorageType(op.getFilter());
    auto filter_i8 = filter_op.read<int8_t>();
    std::vector<int64_t> new_shape = {1, attr.oc, new_kh * new_kw,
                                      align_up(attr.ic, 4l)};
    auto filter_new = std::make_shared<std::vector<int8_t>>(
        attr.oc * new_kh * new_kw * align_up(attr.ic, 4l));
    for (int ioc = 0; ioc < attr.oc; ioc++) {
      for (int iic = 0; iic < attr.ic; iic++) {
        for (int ikh = 0; ikh < attr.kh; ikh++) {
          for (int ikw = 0; ikw < attr.kw; ikw++) {
            int offset =
                ((ioc * input_shape[1] + iic) * attr.kh + ikh) * attr.kw + ikw;
            int new_offset = iic + ikw * factor_dw * align_up(attr.ic, 4) +
                             ikh * factor_dh * new_kw * align_up(attr.ic, 4) +
                             ioc * new_kh * new_kw * align_up(attr.ic, 4);
            filter_new->at(new_offset) = filter_i8->at(offset);
          }
        }
      }
    }
    filter_i8 = filter_new;
    auto new_type = RankedTensorType::get(new_shape, filter_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
    op.setDilationsAttr(rewriter.getI64ArrayAttr({new_dh, new_dw}));
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr({new_kh, new_kw}));
  } else if (attr.is_dw == false) {
    if (op.getUse_3icOptimize()) {
      std::vector<int64_t> new_shape = {attr.ic / attr.groups, attr.oc, attr.kh,
                                        attr.kw};
      new_shape[0] = new_shape[0] * new_shape[2];
      new_shape[2] = 1;
      int ic = new_shape[0];
      int oc = new_shape[1];
      int kh = new_shape[2];
      int kw = new_shape[3];
      int new_count = oc * align_up(ic, 4) * kh * kw;
      auto filter_new = std::make_shared<std::vector<int8_t>>(new_count, 0);
      auto filter_op = op.getFilter().getDefiningOp<top::WeightOp>();
      auto filter_type = module::getStorageType(op.getFilter());
      auto filter_i8 = filter_op.read<int8_t>();
      for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
        for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
          for (int k_idx = 0; k_idx < kh * kw; k_idx++) {
            int orig_offset = ic_idx * kh * kw + k_idx + oc_idx * kh * kw * ic;
            int trans_offset = ic_idx + k_idx * align_up(ic, 4) +
                               oc_idx * kh * kw * align_up(ic, 4);
            filter_new->at(trans_offset) = filter_i8->at(orig_offset);
          }
        }
      }
      new_shape[3] = 1;
      new_shape[2] = kh * kw * align_up(ic, 4);
      new_shape[0] = 1;
      auto new_type = RankedTensorType::get(new_shape, filter_type);
      auto new_filter = top::WeightOp::create(
          op.getFilter().getDefiningOp(), "reorderd", *filter_new, new_type);
      op->setOperand(1, new_filter);
    } else {
      int new_count = new_shape[0] * new_shape[1] * new_shape[2] * new_shape[3];
      auto filter_new = std::make_shared<std::vector<int8_t>>(new_count, 0);
      conv_weight_transform(attr.ic / attr.groups, attr.oc, attr.kh, attr.kw,
                            filter_int8->data(), filter_new->data(),
                            type_bytes);
      auto new_type = RankedTensorType::get(new_shape, filter_type);
      auto new_filter = top::WeightOp::create(
          op.getFilter().getDefiningOp(), "reorderd", *filter_new, new_type);
      op->setOperand(1, new_filter);
    }
  } else {
    int64_t filter_shape[4];
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kh * attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_type);
  }
  if (attr.has_bias) {
    auto bias = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_type = module::getElementType(bias);
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto old_bias = bias.read<int16_t>();
    std::vector<int16_t> new_bias(attr.oc);
    for (int i = 0; i < attr.oc; i++) {
      new_bias[i] = (*old_bias)[i];
    }

    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
    auto new_bias_op = top::WeightOp::create(op.getBias().getDefiningOp(),
                                             "bias", new_bias, new_type);
    op.setOperand(2, new_bias_op);
  }
  return success();
}

/**
 * Reorder twice,
 * first by winograd_weight_transform_subfunc
 * sec in function, merge bias
 */
LogicalResult WinoWeightArr(tpu::Conv2DOp op, PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  int input_c = attr.ic;
  int output_c = attr.oc;
  int groups = attr.groups;

  // read data
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = module::getElementType(op.getFilter());

  auto old_filter_shape = module::getShape(op.getFilter());
  // n, c, 4, 4
  auto num_connection = old_filter_shape[0] * old_filter_shape[1];

  std::vector<int8_t> new_weight(num_connection * 4 * 4);
  auto conv_droped_size = num_connection * 3 * 3;
  winograd_weight_transform_subfunc(
      (char *)filter_int8->data() + conv_droped_size, (char *)new_weight.data(),
      input_c * groups, output_c, groups, 2);

  int new_c = attr.oc < 64 ? attr.oc : 64;
  int new_h = ceiling_func(attr.oc, 64);

  std::vector<int64_t> new_shape(4);
  new_shape[0] = groups;
  new_shape[1] = new_c;
  new_shape[2] = new_h;
  new_shape[3] = ceiling_func(attr.ic, 4) * 64;
  auto new_type = RankedTensorType::get(new_shape, filter_type);
  op.getBias().setType(new_type);
  auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                          "winograd", new_weight, new_type);
  op.setOperand(1, new_filter);
  return success();
}

LogicalResult WinoBiasArr(tpu::Conv2DOp op, PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  int NPU_NUM = 64;
  int groups = attr.groups;
  int oc = attr.oc;
  int occupy_npu_nm = (oc / groups) < NPU_NUM ? (oc / groups) : NPU_NUM;
  int oc_per_npu = ceiling_func(oc / groups, NPU_NUM);

  auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
  auto bias_int8 = biasOp.read<int16_t>();
  auto bias_size = bias_int8->size() / 2;
  auto bias_type = module::getElementType(op.getBias());

  std::vector<int16_t> bias_wino(groups * occupy_npu_nm * oc_per_npu);

  winograd_bias_transform_subfunc(bias_int8->data() + bias_size,
                                  bias_wino.data(), oc, groups);

  std::vector<int64_t> new_shape(4);
  new_shape[0] = 1;
  new_shape[1] = groups * occupy_npu_nm;
  new_shape[2] = 1;
  new_shape[3] = oc_per_npu;
  auto new_type = RankedTensorType::get(new_shape, bias_type);
  auto new_bias = top::WeightOp::create(op.getBias().getDefiningOp(),
                                        "winograd_bias", bias_wino, new_type);
  op.setOperand(2, new_bias);
  return success();
}

/**
 * Reorder twice,
 * first by winograd_weight_transform_subfunc
 * sec in function, merge bias
 */
LogicalResult WinoWeightBiasArr(tpu::Conv2DOp op, PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  int input_c = attr.ic;
  int output_c = attr.oc;
  int groups = attr.groups;

  // read data
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = module::getElementType(op.getFilter());

  auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
  auto bias_int8 = biasOp.read<int16_t>();
  int bias_size = bias_int8->size() / 2;

  auto old_filter_shape = module::getShape(op.getFilter());
  // n, c, 4, 4
  auto num_connection = old_filter_shape[0] * old_filter_shape[1];

  std::vector<int8_t> new_weight_(num_connection * 4 * 4, 0);
  int conv_droped_size = num_connection * 3 * 3;
  winograd_weight_transform_subfunc(
      (char *)filter_int8->data() + conv_droped_size,
      (char *)new_weight_.data(), input_c * groups, output_c, groups, 2);

  int new_c = attr.oc < 64 ? attr.oc : 64;
  int new_h = ceiling_func(attr.oc, 64);

  char *old_weight = (char *)new_weight_.data();
  short *bias_data = (short *)(bias_int8->data() + bias_size);
  std::vector<int8_t> new_weight(
      groups * new_c * new_h * (ceiling_func(attr.ic, 4) * 64 + 2), 0);

  // merge bias
  for (int g = 0; g < groups; g++) {
    for (int m = 0; m < new_c; m++) {
      char *pDst =
          (char *)new_weight.data() +
          (g * new_c + m) * (new_h * ceiling_func(attr.ic, 4) * 64 + 2 * new_h);
      char *pSrc = (char *)old_weight +
                   (g * new_c + m) * (new_h * ceiling_func(attr.ic, 4) * 64);

      memcpy(pDst, pSrc, new_h * ceiling_func(attr.ic, 4) * 64);

      for (int c0 = 0; c0 < new_h; c0++) {
        int oc_idx = c0 * new_c + m;
        if (oc_idx < attr.oc)
          memcpy(pDst + new_h * ceiling_func(attr.ic, 4) * 64,
                 (short *)bias_data + oc_idx + g * attr.oc, 2);
        else
          memset(pDst + new_h * ceiling_func(attr.ic, 4) * 64, 0, 2);
        pDst += 2;
      }
    }
  }
  std::vector<int64_t> new_shape(4);
  new_shape[0] = groups;
  new_shape[1] = new_c;
  new_shape[2] = new_h;
  new_shape[3] = ceiling_func(attr.ic, 4) * 64 + 2;
  auto new_type = RankedTensorType::get(new_shape, filter_type);
  op.getBias().setType(new_type);
  auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                          "winograd", new_weight, new_type);
  op.setOperand(1, new_filter);
  // erase the bias
  op.setOperand(2, module::getNoneOp(op));
  return success();
}

LogicalResult winoReorder(tpu::Conv2DOp op, PatternRewriter &rewriter) {

  // merge weight and bias
  auto attr = op.parseParam();
  bool depthwise =
      (attr.groups == attr.ic && attr.groups == attr.oc && attr.groups > 1);
  if (attr.has_bias && !depthwise) {
    // rewriter.create<top::WeightOp>()
    return WinoWeightBiasArr(op, rewriter);
  } else {
    WinoWeightArr(op, rewriter);
    if (attr.has_bias) {
      WinoBiasArr(op, rewriter);
    }
    return failure();
  }

  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewriteImpl(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  if (module::isWeight(op.getFilter()) == false) {
    return failure();
  }

  if (op.getUseWinograd().value_or(0) == 2) {
    /* decided in ConvOp TopToTpu pass */
    return winoReorder(op, rewriter);
  } else {
    return convReorder(op, rewriter);
  }
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, Float32Type>::matchAndRewriteImpl(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();
  if (module::isWeight(op.getFilter()) == false) {
    return failure();
  }
  auto attr = op.parseParam();
  auto type_bytes = 4;
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto weight_data = filterOp.read_as_byte();
  if (attr.is_dw == false) {
    std::vector<int64_t> new_shape = {1, attr.oc, attr.kh * attr.kw,
                                      align_up(attr.ic / attr.groups, 2l)};
    int new_count =
        align_up(attr.ic / attr.groups, 2l) * attr.oc * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<float>>(new_count, 0);
    conv_weight_transform(attr.ic / attr.groups, attr.oc, attr.kh, attr.kw,
                          weight_data->data(), filter_new->data(), type_bytes);
    auto new_type = RankedTensorType::get(new_shape, out_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  } else {
    int64_t filter_shape[4];
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kh * attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);
  }

  // bias op
  if (attr.has_bias) {
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}
