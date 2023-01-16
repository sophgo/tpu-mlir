//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM1684/WeightReorder.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684;

void conv_weight_transform(int ic, int oc, int kh, int kw,
                           const void *weight_orig, const void *weight_trans,
                           int type_bytes) {
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kh * kw; k_idx++) {
        int orig_offset = ic_idx * kh * kw + k_idx + oc_idx * kh * kw * ic;
        int trans_offset = ic_idx + k_idx * ic + oc_idx * kh * kw * ic;
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

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  auto attr = op.parseParam();
  // if (attr.is_dw == false) {
  //   // merge weight and bias
  //   return success();
  // }
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  int new_size = attr.oc * (align_up(attr.ic, 4l)) * attr.kh * attr.kw;
  auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
  for (int oc_idx = 0; oc_idx < attr.oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < attr.ic; ic_idx++) {
      for (int k_idx = 0; k_idx < attr.kh * attr.kw; k_idx++) {
        int orig_offset = ic_idx * attr.kh * attr.kw + k_idx +
                          oc_idx * attr.kh * attr.kw * attr.ic;
        int trans_offset = ic_idx + k_idx * align_up(attr.ic, 4l) +
                           oc_idx * (attr.kh * attr.kw * align_up(attr.ic, 4l));
        filter_new->at(trans_offset) = filter_int8->at(orig_offset);
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> new_shape = {
      1, attr.oc, attr.kh * attr.kw * align_up(attr.ic, 4l), 1};
  auto new_type =
      RankedTensorType::get(new_shape, filter_type.getElementType());
  auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                          "reorderd", *filter_new, new_type);
  op->setOperand(1, new_filter);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, Float32Type>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();
  auto type_bytes = 4;
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto weight_data = filterOp.read_as_byte();
  if (attr.is_dw == false) {
    std::vector<int64_t> new_shape = {
        1, attr.oc, attr.kh * attr.kw * align_up(attr.ic / attr.groups, 4l), 1};
    int new_count =
        align_up(attr.ic / attr.groups, 4l) * attr.oc * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<float>>(new_count, 0);
    conv_weight_transform(attr.ic, attr.oc, attr.kh, attr.kw,
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
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}

void tpu::Conv2DOp::codegen_global_bm1684() {
  auto attr = parseParam();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  auto filter_addr = module::getAddress(getFilter());
  auto bias_addr = module::getAddress(getBias());
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    if (attr.is_dw) {
      BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl, attr.pwr,
          attr.sh, attr.sw, attr.ins_h, attr.ins_w, shift,
          attr.has_bias ? 1 : 0, 0, 1, 1, 1, 1, attr.do_relu ? 1 : 0,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      BM1684::instance()
          .dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
              in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
              attr.ih, attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh,
              attr.dw, attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
              attr.has_bias ? 1 : 0, 0, attr.do_relu ? 1 : 0, 0, 1, 0, 0, shift,
              1, 1, 1, 3, 0, 0, 0, 0, 0,
              (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    }
  } else {
    // F32
    if (attr.is_dw) {
      BM1684::instance().dl_nodechip_depthwise_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl, attr.pwr,
          attr.sh, attr.sw, attr.dh, attr.dw, attr.has_bias ? 1 : 0,
          attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      BM1684::instance().dl_nodechip_conv_forward_parallel_with_data_split(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh, attr.dw,
          attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
          attr.has_bias ? 1 : 0, 0 /* result_add*/, attr.do_relu ? 1 : 0,
          attr.relu_limit, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    }
  }
}

int64_t tpu::Conv2DOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::Conv2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not Implemented");
}
