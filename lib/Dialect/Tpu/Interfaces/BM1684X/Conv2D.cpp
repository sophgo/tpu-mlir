//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ConvUtils.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynamicLayer.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

// ======================================
// WeightReorderInterface
// ======================================

// refer to net_compiler: bool BM1684XCoeffArranger::ConvWeightArr(GraphEdge*
// edge)
template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8) ||
      op.getCoeffMerged())
    return failure();

  auto attr = op.parseParam();
  int input_c = attr.ic;
  int output_c = attr.oc;
  int kh = attr.kh;
  int kw = attr.kw;
  int stride_h = attr.sh;
  int stride_w = attr.sw;
  int groups = attr.groups;
  int gic = input_c / groups;

  bool strideh_gt_15 = stride_h > 15;
  bool stridew_gt_15 = stride_w > 15;
  bool stride_hw_gt_15 = strideh_gt_15 || stridew_gt_15;
  int cell_h = kh, cell_w = kw;
  int IC_PARALLEL = BM168x::ic_num(1);

  if (strideh_gt_15) {
    for (int i = 15; i > 1; i--) {
      if (kh % i == 0) {
        cell_h = i;
        break;
      }
    }
  }

  if (stridew_gt_15) {
    for (int i = 15; i > 1; i--) {
      if (kw % i == 0) {
        cell_w = i;
        break;
      }
    }
  }

  bool merge = true;
  bool merge_with_requant = true;
  auto out_stype = module::getStorageType(op.getOutput());
  if (out_stype.isInteger(32)) {
    merge_with_requant = false;
    if (attr.has_bias == false) merge = false;
  }
  bool isINT4Conv = false;
  auto in_stype = module::getStorageType(op.getInput());
  isINT4Conv = (in_stype.isInteger(4) && !attr.is_dw);
  if (isINT4Conv) {
    IC_PARALLEL = BM168x::ic_num(0.5);
  }

  // filter
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh,
                                       attr.kw};
  auto filter_type = module::getStorageType(op.getFilter());
  int use_3ic_optimize = 0;
  if (attr.ic * attr.kh * attr.kw <= IC_PARALLEL && attr.kh > 1 &&
      attr.kw > 1) {
    use_3ic_optimize = 3; // merge kh and kw to ic
  } else if (attr.ic * attr.kw <= IC_PARALLEL && attr.kw > 1 &&
             (attr.kh < attr.kw || attr.ic * attr.kh > IC_PARALLEL)) {
    use_3ic_optimize = 2; // merge kw to ic
  } else if (attr.ic * attr.kh <= IC_PARALLEL && attr.kh > 1) {
    use_3ic_optimize = 1; // merge kh to ic
  } else {
    use_3ic_optimize = 0;
  }

  auto stype = module::getStorageType(op.getFilter());
  auto new_type = RankedTensorType::get(filter_shape, stype);
  int weight_size = align_up(gic, IC_PARALLEL) * output_c * kh * kw * 1;
  auto data_i8 = std::make_shared<std::vector<int8_t>>(weight_size);

  auto pre_op = op.getInput().getDefiningOp();
  if (use_3ic_optimize && !isa<top::InputOp>(*pre_op)) {
    // broadcast input using BDC rather than GDMA
    use_3ic_optimize |= 0x10;
  }
  if (use_3ic_optimize && !op.getInput().hasOneUse()) {
    // broadcast input using BDC to a buffer
    use_3ic_optimize |= 0x30;
  }
  if (module::isBM1686()) {
    use_3ic_optimize = 0;
  }

  if (attr.is_dw == false) {
    if (stride_hw_gt_15) {
      filter_shape[0] = 1;
      filter_shape[1] = output_c;
      filter_shape[2] = ceiling_func(gic, IC_PARALLEL);
      filter_shape[3] = kh * kw * IC_PARALLEL;

      for (int oc = 0; oc < output_c; oc++) {
        for (int ic_idx = 0; ic_idx < ceiling_func(gic, IC_PARALLEL);
             ic_idx++) {
          for (int ic_inner = 0; ic_inner < IC_PARALLEL; ic_inner++) {
            for (int icell_h = 0; icell_h < (kh / cell_h); icell_h++) {
              for (int ih = 0; ih < cell_h; ih++) {
                for (int icell_w = 0; icell_w < (kw / cell_w); icell_w++) {
                  for (int iw = 0; iw < cell_w; iw++) {
                    if (ic_idx * IC_PARALLEL + ic_inner >= gic)
                      continue;
                    int orig_offset =
                        oc * gic * kh * kw +
                        (ic_idx * IC_PARALLEL + ic_inner) * kh * kw +
                        (icell_h * cell_h + ih) * kw + icell_w * cell_w + iw;
                    int trans_offset =
                        oc * kh * kw * align_up(gic, IC_PARALLEL) +
                        (icell_h * (kw / cell_w) + icell_w) * cell_h * cell_w *
                            align_up(gic, IC_PARALLEL) +
                        ic_idx * cell_h * cell_w * IC_PARALLEL +
                        (ih * cell_w + iw) * IC_PARALLEL + ic_inner;
                    data_i8->at(trans_offset) = filter_i8->at(orig_offset);
                  }
                }
              }
            }
          }
        }
      }
    } else {
      tpu::reshape_coeff_for_3ic(filter_i8, filter_shape, use_3ic_optimize,
                                 isINT4Conv);
      op->setAttr("use_3ic_optimize",
                  rewriter.getI64IntegerAttr(use_3ic_optimize));
    }
  } else {
    filter_shape = {1, attr.oc, 1, attr.kh * attr.kw};
  }

  auto filter_data = (stride_hw_gt_15 == true) ? data_i8 : filter_i8;
  tpu::reshape_coeff_for_broadcast_channel(filter_data, filter_shape, false,
                                           isINT4Conv);
  if (isINT4Conv) {
    tpu::compact_coeff_for_int4(filter_data, filter_shape);
  }

  int64_t new_oc = filter_shape[1];
  int64_t filter_w_bytes = filter_shape[3] * sizeof(int8_t);

  // bias
  i32_array_t bias_new;
  std::vector<int64_t> bias_shape = {1, attr.oc, 1, 1};
  int64_t bias_w_bytes = 0;
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    bias_new = biasOp.read<int32_t>();
    tpu::reshape_coeff_for_broadcast_channel(bias_new, bias_shape, false,
                                             isINT4Conv);
    assert(new_oc == bias_shape[1]);
    bias_w_bytes = bias_shape[3] * sizeof(int32_t);
  }

  // requant
  int64_t quant_w_bytes = 0;
  std::vector<int64_t> quant_shape;
  std::shared_ptr<std::vector<int32_t>> quant_data = nullptr;
  if(merge_with_requant){
    auto qtype = module::getUniformQuantizedType(op.getOutput());
    int32_t out_zp = qtype.getZeroPoint();
    quant_data = std::make_shared<std::vector<int32_t>>(attr.oc * 3, 0);
    auto m_data = module::getI64Array(op.getMultiplier(), attr.oc, 1);
    auto r_data = module::getI64Array(op.getRshift(), attr.oc, 0);
    int64_t quant_w_size = 0;
    bool align = true;
    if (module::isBM1686()) {
      align = false;
      quant_w_size = 2;
      for (int i = 0; i < attr.oc; i++) {
        quant_data->at(i * 2) = m_data->at(i);
        quant_data->at(i * 2 + 1) =
            (int32_t)(((-(int32_t)r_data->at(i)) & 0x000000ff) |
                      ((out_zp & 0x0000ffff) << 16));
      }
    } else {
      quant_w_size = 3;
      for (int i = 0; i < attr.oc; i++) {
        quant_data->at(i * 3) = m_data->at(i);
        quant_data->at(i * 3 + 1) = -r_data->at(i);
        quant_data->at(i * 3 + 2) = out_zp;
      }
    }
    quant_shape = {1, attr.oc, 1, quant_w_size};
    tpu::reshape_coeff_for_broadcast_channel(quant_data, quant_shape, align,
                                             isINT4Conv);
    assert(new_oc == quant_shape[1]);
    quant_w_bytes = quant_shape[3] * sizeof(int32_t);
  }

  // merge
  int64_t quant_offset = 0, bias_offset = 0, filter_offset = 0;
  int64_t filter_align = BM168x::EU_BYTES;
  if (attr.is_dw) {
    if (!module::isBM1686()) {
      filter_align = 1;
    }
  }

  if (attr.has_bias) {
    bias_offset =
        align_up(quant_offset + quant_w_bytes, (int64_t)sizeof(int32_t));
    filter_offset = align_up(bias_offset + bias_w_bytes, filter_align);
  } else {
    filter_offset = align_up(quant_offset + quant_w_bytes, filter_align);
  }
  int64_t merge_w = filter_offset + filter_w_bytes;
  // merge requant/bias/filter
  auto new_coeff = std::make_shared<std::vector<int8_t>>(new_oc * merge_w, 0);
  std::vector<int64_t> coeff_shape = {1, new_oc, 1, merge_w};
  if (isINT4Conv)
    coeff_shape[3] <<= 1;
  for (int i = 0; i < new_oc; i++) {
    auto coeff_ptr = new_coeff->data() + i * merge_w;
    auto bias_ptr =
        attr.has_bias ? (bias_new->data() + i * bias_shape[3]) : nullptr;
    auto filter_ptr = filter_data->data() + i * filter_shape[3];
    // copy quant
    if(merge_with_requant) {
      auto quant_ptr = quant_data->data() + i * quant_shape[3];
      memcpy(coeff_ptr + quant_offset, quant_ptr, quant_w_bytes);
    }
    if (attr.has_bias) {
      memcpy(coeff_ptr + bias_offset, bias_ptr, bias_w_bytes);
    }
    memcpy(coeff_ptr + filter_offset, filter_ptr, filter_w_bytes);
  }
  if (merge_w > MAX_TPU_DIM || coeff_shape[3] > MAX_TPU_DIM) {
    if (attr.is_dw) {
      coeff_shape[2] = ceiling_func(attr.oc, (int64_t)IC_PARALLEL);
      coeff_shape[3] /= coeff_shape[2];
    } else {
      coeff_shape[2] = IC_PARALLEL;
      coeff_shape[3] /= IC_PARALLEL;
    }
  }
  auto elem_type = module::getStorageType(op.getFilter());
  auto coeff_type = RankedTensorType::get(coeff_shape, elem_type);
  bool sign = coeff_type.getElementType().isSignedInteger();
  if (isINT4Conv) {
    coeff_type =
        RankedTensorType::get(coeff_shape, rewriter.getIntegerType(4, sign));
  }
  auto coeff_op = top::WeightOp::create(op, "merge", *new_coeff, coeff_type);
  op->removeAttr("rshift");
  op->removeAttr("multiplier");
  op->setAttr("coeff_merged", rewriter.getBoolAttr(merge));
  op->setOperand(1, coeff_op);
  auto none = module::getNoneOp(op);
  op->setOperand(2, none.getResult());

  auto new_type_ = RankedTensorType::get(coeff_shape, in_stype);
  if (isINT4Conv) {
    op.getFilter().setType(coeff_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::Conv2DOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();

  int input_c = attr.ic;
  int output_c = attr.oc;
  int kh = attr.kh;
  int kw = attr.kw;
  int stride_h = attr.sh;
  int stride_w = attr.sw;
  int groups = attr.groups;
  int gic = input_c / groups;

  int npu_num = BM168x::NPU_NUM;
  const int IC_PARALLEL = BM168x::ic_num(2);
  // int weight_size = align_up(output_c, npu_num) * gic * kh * kw;
  // auto data_f32 = std::make_shared<std::vector<float>>(weight_size);
  // auto out_type = module::getStorageType(op.getOutput());

  bool strideh_gt_15 = stride_h > 15;
  bool stridew_gt_15 = stride_w > 15;
  int cell_h = kh, cell_w = kw;

  if (strideh_gt_15) {
    for (int i = 15; i > 1; i--) {
      if (kh % i == 0) {
        cell_h = i;
        break;
      }
    }
  }

  if (stridew_gt_15) {
    for (int i = 15; i > 1; i--) {
      if (kw % i == 0) {
        cell_w = i;
        break;
      }
    }
  }

  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh,
                                       attr.kw};
  auto filter_u16 = filterOp.read<uint16_t>();
  auto filter_type = module::getStorageType(op.getFilter());
  int weight_size = align_up(gic, IC_PARALLEL) * output_c * kh * kw * 2;
  auto data_bf16 = std::make_shared<std::vector<uint16_t>>(weight_size);

  if (attr.is_dw) {
    filter_shape = {1, attr.ic, attr.kh, attr.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_filter_type);
    if (attr.has_bias) {
      auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
      auto data_fp32 = biasOp.read<float>();
      auto count = data_fp32->size();
      auto data_u16 = std::make_shared<std::vector<uint16_t>>(count);

      bool isF16 = filter_type.isF16();
      for (uint32_t i = 0; i < count; i++) {
        data_u16->at(i) = isF16 ? f32_to_f16(data_fp32->at(i))
                                : f32_to_bf16(data_fp32->at(i));
      }

      int64_t bias_shape[4] = {1, attr.oc, 1, 1};
      auto new_bias_type = RankedTensorType::get(bias_shape, filter_type);
      op.getBias().setType(new_bias_type);

      auto newBiasOp =
          top::WeightOp::create(op, "reordered", *data_u16, new_bias_type);
      op->setOperand(2, newBiasOp);
    }
  } else {
    int use_3ic_optimize = 0;
    if (false) { // Shut down 3ic optimization temporarily for fp16/bfp16
      if (attr.ic * attr.kh * attr.kw <= IC_PARALLEL && attr.kh > 1 &&
          attr.kw > 1) {
        use_3ic_optimize = 3; // merge kh and kw to ic
      } else if (attr.ic * attr.kw <= IC_PARALLEL && attr.kw > 1 &&
                 (attr.kh < attr.kw || attr.ic * attr.kh > IC_PARALLEL)) {
        use_3ic_optimize = 2; // merge kw to ic
      } else if (attr.ic * attr.kh <= IC_PARALLEL && attr.kh > 1) {
        use_3ic_optimize = 1; // merge kh to ic
      } else {
        use_3ic_optimize = 0;
      }
      if (use_3ic_optimize) {
        // Now only support broadcast using BDC when it is a local layer.
        use_3ic_optimize |= 0x10;
      }
    }

    /////////////// this branch is speical for stride > 15
    if (strideh_gt_15 || stridew_gt_15) {
      vector<int> cell_h;
      vector<int> cell_w;
      vector<int> cell_h_sum;
      vector<int> cell_w_sum;
      int cell_num_h = 1;
      int cell_num_w = 1;
      int max_cell_h = kh;
      int max_cell_w = kw;
      // split kernel
      if (strideh_gt_15) {
        cell_num_h = ceiling_func(kh, 15);
        max_cell_h = ceiling_func(kh, cell_num_h);
        int cur_h = 0;
        int sum_h = 0;
        for (int i = 0; i < cell_num_h; i++) {
          cur_h = kh / cell_num_h + ((i < kh % cell_num_h) ? 1 : 0);
          cell_h.push_back(cur_h);
          cell_h_sum.push_back(sum_h);
          sum_h += cur_h;
        }
      } else {
        cell_h.push_back(max_cell_h);
        cell_h_sum.push_back(0);
      }
      if (stridew_gt_15) {
        cell_num_w = ceiling_func(kw, 15);
        max_cell_w = ceiling_func(kw, cell_num_w);
        int cur_w = 0;
        int sum_w = 0;
        for (int i = 0; i < cell_num_w; i++) {
          cur_w = kw / cell_num_w + ((i < kw % cell_num_w) ? 1 : 0);
          cell_w.push_back(cur_w);
          cell_w_sum.push_back(sum_w);
          sum_w += cur_w;
        }
      } else {
        cell_w.push_back(max_cell_w);
        cell_w_sum.push_back(0);
      }
      int oc_per_groups = output_c / groups;
      int weight_size_per_group =
          ((oc_per_groups < npu_num) ? oc_per_groups
                                     : align_up(oc_per_groups, npu_num)) *
          align_up(gic, IC_PARALLEL) * cell_num_h * max_cell_h * cell_num_w *
          max_cell_w;
      weight_size = groups * weight_size_per_group;
      data_bf16->resize(weight_size, 0);
      // Must be initialized to 0. It is to avoid memory increase when bmodel
      // combine.
      int ocloops = ceiling_func(oc_per_groups, npu_num);
      for (int group_idx = 0; group_idx < groups; group_idx++) {
        for (int oc = 0; oc < oc_per_groups; oc++) {
          for (int ic_idx = 0; ic_idx < ceiling_func(gic, IC_PARALLEL);
               ic_idx++) {
            for (int ic_inner = 0; ic_inner < IC_PARALLEL; ic_inner++) {
              for (int cell_h_idx = 0; cell_h_idx < cell_num_h; cell_h_idx++) {
                for (int ih = 0; ih < cell_h[cell_h_idx]; ih++) {
                  for (int cell_w_idx = 0; cell_w_idx < cell_num_w;
                       cell_w_idx++) {
                    for (int iw = 0; iw < cell_w[cell_w_idx]; iw++) {
                      if (ic_idx * IC_PARALLEL + ic_inner >= gic)
                        continue;
                      int orig_offset =
                          group_idx * oc_per_groups * gic * kh * kw +
                          oc * gic * kh * kw +
                          (ic_idx * IC_PARALLEL + ic_inner) * kh * kw +
                          cell_h_sum[cell_h_idx] * kw + ih * kw +
                          cell_w_sum[cell_w_idx] + iw;
                      int trans_offset =
                          groups * (oc % npu_num) * ocloops *
                              align_up(gic, IC_PARALLEL) * cell_num_h *
                              max_cell_h * cell_num_w * max_cell_w + // npu idx
                          group_idx * ocloops * align_up(gic, IC_PARALLEL) *
                              cell_num_h * max_cell_h * cell_num_w *
                              max_cell_w + // group idx
                          (cell_h_idx * cell_num_w + cell_w_idx) * ocloops *
                              max_cell_h * max_cell_w *
                              align_up(gic, IC_PARALLEL) + // cell idx
                          (oc / npu_num) * cell_h[cell_h_idx] *
                              cell_w[cell_w_idx] *
                              align_up(gic, IC_PARALLEL) + // oc offset
                          ic_idx * IC_PARALLEL * cell_h[cell_h_idx] *
                              cell_w[cell_w_idx] + // ic idx
                          (ih * cell_w[cell_w_idx] + iw) * IC_PARALLEL +
                          ic_inner;
                      data_bf16->at(trans_offset) = filter_u16->at(orig_offset);
                    }
                  }
                }
              }
            }
          }
        }
      }
      filter_shape[0] = 1;
      filter_shape[1] = (oc_per_groups < npu_num) ? oc_per_groups : npu_num;
      filter_shape[2] = 1;
      filter_shape[3] = groups * ocloops * align_up(gic, IC_PARALLEL) *
                        cell_num_h * max_cell_h * cell_num_w * max_cell_w;
      if (filter_shape[3] > MAX_TPU_DIM) {
        if (attr.is_dw) {
          filter_shape[2] = ceiling_func(attr.oc, (int64_t)IC_PARALLEL);
          filter_shape[3] /= filter_shape[2];
        } else {
          filter_shape[2] = IC_PARALLEL;
          filter_shape[3] /= IC_PARALLEL;
        }
      }
    } else {
      tpu::reshape_coeff_for_3ic(filter_u16, filter_shape, use_3ic_optimize);
      op->setAttr("use_3ic_optimize",
                  rewriter.getI64IntegerAttr(use_3ic_optimize));
    }
    auto new_type = RankedTensorType::get(filter_shape, filter_type);
    auto filter_data =
        (strideh_gt_15 || stridew_gt_15) == true ? *data_bf16 : *filter_u16;
    auto new_op =
        top::WeightOp::create(op, "filter_reorderd", filter_data, new_type);
    op->setOperand(1, new_op);
    // bias op
    if (attr.has_bias) {
      auto bias_type = module::getStorageType(op.getBias());
      int64_t bias_shape[4] = {1, attr.oc, 1, 1};
      auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
      op.getBias().setType(new_bias_type);
    }
    return success();
  }
  return failure();
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, BFloat16Type>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, Float16Type>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, Float32Type>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32()) {
    return failure();
  }
  if (module::isWeight(op.getFilter()) == false) {
    return failure();
  }
  auto attr = op.parseParam();
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_f32 = filterOp.read<float>();
  auto filter_type = module::getStorageType(op.getFilter());
  int input_c = attr.ic;
  int output_c = attr.oc;
  int kh = attr.kh;
  int kw = attr.kw;
  int stride_h = attr.sh;
  int stride_w = attr.sw;
  int groups = attr.groups;
  int gic = input_c / groups;
  bool strideh_gt_15 = stride_h > 15;
  bool stridew_gt_15 = stride_w > 15;
  int cell_h = kh, cell_w = kw;
  int npu_num = BM168x::NPU_NUM;
  const int IC_PARALLEL = BM168x::ic_num(4);
  int weight_size = align_up(output_c, npu_num) * gic * kh * kw;
  auto data_f32 = std::make_shared<std::vector<float>>(weight_size);
  auto out_type = module::getStorageType(op.getOutput());

  if (strideh_gt_15) {
    for (int i = 15; i > 1; i--) {
      if (kh % i == 0) {
        cell_h = i;
        break;
      }
    }
  }

  if (stridew_gt_15) {
    for (int i = 15; i > 1; i--) {
      if (kw % i == 0) {
        cell_w = i;
        break;
      }
    }
  }

  // filter reorder
  std::vector<int64_t> filter_shape = {1, output_c, gic, kh * kw};
  if (out_type.isF32()) {
    if (strideh_gt_15 || stridew_gt_15) {
      vector<int> cell_h;
      vector<int> cell_w;
      vector<int> cell_h_sum;
      vector<int> cell_w_sum;
      int cell_num_h = 1;
      int cell_num_w = 1;
      int max_cell_h = kh;
      int max_cell_w = kw;

      if (strideh_gt_15) {
        cell_num_h = ceiling_func(kh, 15);
        max_cell_h = ceiling_func(kh, cell_num_h);
        int cur_h = 0;
        int sum_h = 0;
        for (int i = 0; i < cell_num_h; i++) {
          cur_h = kh / cell_num_h + ((i < kh % cell_num_h) ? 1 : 0);
          cell_h.push_back(cur_h);
          cell_h_sum.push_back(sum_h);
          sum_h += cur_h;
        }
      } else {
        cell_h.push_back(max_cell_h);
        cell_h_sum.push_back(0);
      }

      if (stridew_gt_15) {
        cell_num_w = ceiling_func(kw, 15);
        max_cell_w = ceiling_func(kw, cell_num_w);
        int cur_w = 0;
        int sum_w = 0;
        for (int i = 0; i < cell_num_w; i++) {
          cur_w = kw / cell_num_w + ((i < kw % cell_num_w) ? 1 : 0);
          cell_w.push_back(cur_w);
          cell_w_sum.push_back(sum_w);
          sum_w += cur_w;
        }
      } else {
        cell_w.push_back(max_cell_w);
        cell_w_sum.push_back(0);
      }

      int oc_per_groups = output_c / groups;
      int weight_size_per_group =
          ((oc_per_groups < npu_num) ? oc_per_groups
                                     : align_up(oc_per_groups, npu_num)) *
          gic * cell_num_h * max_cell_h * cell_num_w * max_cell_w;
      size_t weight_size = groups * weight_size_per_group;
      auto data_f32 = std::make_shared<std::vector<float>>(weight_size);
      int ocloops = ceiling_func(oc_per_groups, npu_num);
      for (int group_idx = 0; group_idx < groups; group_idx++) {
        for (int oc = 0; oc < oc_per_groups; oc++) {
          for (int ic = 0; ic < gic; ic++) {
            for (int cell_h_idx = 0; cell_h_idx < cell_num_h; cell_h_idx++) {
              for (int ih = 0; ih < cell_h[cell_h_idx]; ih++) {
                for (int cell_w_idx = 0; cell_w_idx < cell_num_w;
                     cell_w_idx++) {
                  for (int iw = 0; iw < cell_w[cell_w_idx]; iw++) {
                    int orig_offset =
                        group_idx * oc_per_groups * gic * kh * kw +
                        oc * gic * kh * kw + ic * kh * kw +
                        cell_h_sum[cell_h_idx] * kw + ih * kw +
                        cell_w_sum[cell_w_idx] + iw;
                    int trans_offset =
                        groups * (oc % npu_num) * ocloops * gic * cell_num_h *
                            max_cell_h * cell_num_w * max_cell_w + // npu idx
                        group_idx * ocloops * gic * cell_num_h * max_cell_h *
                            cell_num_w * max_cell_w + // group idx
                        (cell_h_idx * cell_num_w + cell_w_idx) * ocloops *
                            max_cell_h * max_cell_w * gic + // cell idx
                        (oc / npu_num) * cell_h[cell_h_idx] *
                            cell_w[cell_w_idx] * gic + // oc offset
                        ic * cell_h[cell_h_idx] * cell_w[cell_w_idx] +
                        ih * cell_w[cell_w_idx] + iw;
                    data_f32->at(trans_offset) = filter_f32->at(orig_offset);
                  }
                }
              }
            }
          }
        }
      }

      filter_shape[0] = 1;
      filter_shape[1] = (oc_per_groups < npu_num) ? oc_per_groups : npu_num;
      filter_shape[2] = 1;
      filter_shape[3] = groups * ocloops * gic * cell_num_h * max_cell_h *
                        cell_num_w * max_cell_w;

      if (filter_shape[3] > MAX_TPU_DIM) {
        if (attr.is_dw) {
          filter_shape[2] = ceiling_func(attr.oc, (int64_t)IC_PARALLEL);
          filter_shape[3] /= filter_shape[2];
        } else {
          filter_shape[2] = IC_PARALLEL;
          filter_shape[3] /= IC_PARALLEL;
        }
      }
      auto new_type = RankedTensorType::get(filter_shape, out_type);
      op.getFilter().setType(new_type);
      auto new_op =
          top::WeightOp::create(op, "filter_reorderd", *data_f32, new_type);
      op->setOperand(1, new_op);
    } else {
      auto new_type = RankedTensorType::get(filter_shape, out_type);
      op.getFilter().setType(new_type);
    }
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Conv2DOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (attr.dims == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  conv_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto &common = spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = attr.pht;
  common.pad_h_b = attr.phb;
  common.pad_w_l = attr.pwl;
  common.pad_w_r = attr.pwr;
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    auto out_etype = module::getStorageType(getOutput());
    if (out_etype.isUnsignedInteger()) {
      common.if_relu = true;
    }
    if (out_etype.isInteger(32)){
      bool coeff_merge = getCoeffMerged();
      spec.merge_coeff = coeff_merge ? 1 : 0;
    } else {
      spec.merge_coeff = 2;
    }
    common.is_asym = true;
    common.ipad_value = in_qtype.getZeroPoint();
  }
  BM168x::call_global_func("backend_api_conv_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv2DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t in_dslice, int64_t in_wslice, int64_t out_nslice,
    int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  if (module::isBM1686() && getCoeffMerged()) {
    return 0;
  }
  auto &p = getConv2DParam(*this);
  int64_t sz = 0;
  auto in_type = BM168x::getDataType(getInput());
  auto out_type = BM168x::getDataType(getOutput());
  auto in_type_len = BM168x::getFmtBytes(in_type);
  auto out_type_len = BM168x::getFmtBytes(out_type);
  auto eu_num = BM168x::eu_num(in_type_len);
  int oc_per_npu = ceiling_func(p.oc, BM168x::NPU_NUM);
  int ic_per_npu = ceiling_func(p.ic / p.groups, BM168x::NPU_NUM);
  int int32_size = out_lmem_bytes * sizeof(int32_t) / out_type_len;
  if (getCoeffMerged()) {
    sz += int32_size;
  }
  if (p.groups > 1) {
    sz += in_nslice * ic_per_npu * align_up(in_hslice * in_wslice, eu_num) *
          in_type_len;
    sz += ic_per_npu * 2 * in_type_len;
  }

  if (p.is_dw) {
    sz += int32_size;
    sz += oc_per_npu * p.kh * p.kw;
  }

  if (getUse_3icOptimize() & 0x20) {
    // used for broadcast input
    sz += in_lmem_bytes;
  }
  int use_3ic = (getUse_3icOptimize() & 0x3);
  if (use_3ic == 1) { // merge kh to ic
    sz += align_up(out_hslice * in_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kh * in_type_len;
  } else if (use_3ic == 2) { // merge kw to ic
    sz += align_up(in_hslice * out_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kw * in_type_len;
  } else if (use_3ic == 3) { // merge kh and kw to ic
    sz += align_up(out_hslice * out_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kh * p.kw * in_type_len;
  }
  return sz;
}

void tpu::Conv2DOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                          int64_t d_step, int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (attr.dims == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step);

  conv_local_param_t p;
  memset(&p, 0, sizeof(p));
  p.spec.buffer_local_addr = gi.buffer_addr;
  auto &common = p.spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = (in_gi.h_idx == 0 ? attr.pht : 0);
  common.pad_h_b = (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.phb : 0);
  common.pad_w_l = (in_gi.w_idx == 0 ? attr.pwl : 0);
  common.pad_w_r = (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pwr : 0);
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    common.ipad_value = in_qtype.getZeroPoint();
    common.is_asym = true;
    auto out_etype = module::getStorageType(getOutput());
    common.if_relu = out_etype.isUnsignedInteger();
    if (out_etype.isInteger(32)){
      bool coeff_merge = getCoeffMerged();
      p.spec.merge_coeff = coeff_merge ? 1 : 0;
      p.spec.with_requant = 0;
    } else {
      p.spec.merge_coeff = 2;
      p.spec.with_requant = 1;
    }
  }
  BM168x::call_local_func("backend_api_conv_local", &p, sizeof(p), &sec_info,
                          input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::Conv2DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(conv_local_param_t);
  conv_local_param_t param;
  memset(&param, 0, sizeof(param));
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (attr.dims == 1) {
    BM168x::fix_shape(input_spec->at(0), {attr.n, attr.ic, attr.ih, attr.iw});
    BM168x::fix_shape(output_spec->at(0), {attr.n, attr.oc, attr.oh, attr.ow});
  }
  auto gi = getGroupInfo(0, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);

  param.spec.buffer_local_addr = gi.buffer_addr;
  auto &common = param.spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = attr.pht;
  common.pad_h_b = attr.phb;
  common.pad_w_l = attr.pwl;
  common.pad_w_r = attr.pwr;
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    common.ipad_value = in_qtype.getZeroPoint();
    common.is_asym = true;
    auto out_etype = module::getStorageType(getOutput());
    common.if_relu = out_etype.isUnsignedInteger();
    if (out_etype.isInteger(32)){
      bool coeff_merge = getCoeffMerged();
      param.spec.merge_coeff = coeff_merge ? 1 : 0;
      param.spec.with_requant = 0;
    } else {
      param.spec.merge_coeff = 2;
      param.spec.with_requant = 1;
    }
  }
  param.spec.reference_id = get_tensor_id(op->getResult(0));
  param.spec.concat_c = attr.oc;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Conv2DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(conv_global_spec_t);
  conv_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto attr = parseParam();
  auto &common = spec.common;
  common.input_c = attr.ic;
  common.output_c = attr.oc;
  common.if_relu = attr.do_relu;
  common.upper_limit = attr.relu_limit;
  common.kh = attr.kh;
  common.kw = attr.kw;
  common.dh = attr.dh;
  common.dw = attr.dw;
  common.stride_h = attr.sh;
  common.stride_w = attr.sw;
  common.groups = attr.groups;
  common.pad_h_t = attr.pht;
  common.pad_h_b = attr.phb;
  common.pad_w_l = attr.pwl;
  common.pad_w_r = attr.pwr;
  common.round_mode = ROUNDING_HALF_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  common.use_3ic_optimize = getUse_3icOptimize();
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    common.ipad_value = in_qtype.getZeroPoint();
    common.is_asym = true;
    auto out_etype = module::getStorageType(getOutput());
    common.if_relu = out_etype.isUnsignedInteger();
    if (out_etype.isInteger(32)){
      bool coeff_merge = getCoeffMerged();
      spec.merge_coeff = coeff_merge ? 1 : 0;
    } else {
      spec.merge_coeff = 2;
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::Conv2DOp::get_fw_type_bm1684x() { return FW_BMNET_CONV; }
