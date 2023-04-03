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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684;

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

template <>
LogicalResult WeightReorder<tpu::Conv2DOp, int8_t>::matchAndRewrite(
    tpu::Conv2DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  auto attr = op.parseParam();
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = module::getElementType(op.getFilter());
  if (attr.is_dw == false) {
    int new_size = attr.oc * (align_up(attr.ic, 4l)) * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
    for (int oc_idx = 0; oc_idx < attr.oc; oc_idx++) {
      for (int ic_idx = 0; ic_idx < attr.ic; ic_idx++) {
        for (int k_idx = 0; k_idx < attr.kh * attr.kw; k_idx++) {
          int orig_offset = ic_idx * attr.kh * attr.kw + k_idx +
                            oc_idx * attr.kh * attr.kw * attr.ic;
          int trans_offset =
              ic_idx + k_idx * align_up(attr.ic, 4l) +
              oc_idx * (attr.kh * attr.kw * align_up(attr.ic, 4l));
          filter_new->at(trans_offset) = filter_int8->at(orig_offset);
        }
      }
    }
    std::vector<int64_t> new_shape = {
        1, attr.oc, attr.kh * attr.kw * align_up(attr.ic, 4l), 1};
    auto new_type = RankedTensorType::get(new_shape, filter_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);
  } else {
    int64_t filter_shape[4];
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kh * attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_type);
  }
  // bias op
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_type = module::getElementType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
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
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = attr.has_bias ? module::isSign(getBias()) : 0;
    auto out_sign = module::isSign(getOutput());
    if (attr.is_dw) {
      BM1684::instance().dl_nodechip_depthwise_fix8b_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, attr.pht, attr.phb, attr.pwl, attr.pwr,
          attr.sh, attr.sw, attr.ins_h, attr.ins_w, shift,
          attr.has_bias ? 1 : 0, /*shift_sign*/ 0, in_sign, filter_sign,
          bias_sign, out_sign, attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      BM1684::instance()
          .dl_nodechip_conv_forward_parallel_fix8b_with_data_split(
              in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic,
              attr.ih, attr.iw, attr.groups, attr.oc, attr.kh, attr.kw, attr.dh,
              attr.dw, attr.pht, attr.phb, attr.pwl, attr.pwr, attr.sh, attr.sw,
              attr.has_bias ? 1 : 0, 0, attr.do_relu ? 1 : 0, 0, 1, 0, 0, shift,
              in_sign, filter_sign, bias_sign, 3, 0, 0, 0, 0, 0,
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
  return 0;
}

void tpu::Conv2DOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                         local_sec_info_t &sec_info) {
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto f_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  auto gi = getGroupInfo(n_step, h_step, 0, 0);
  auto p = parseParam();
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.ic, (int)in_gi.h_slice,
                       (int)p.iw};
  int top_dim[4] = {(int)gi.n_slice, (int)p.oc, (int)gi.h_slice, (int)p.ow};
  auto pad_h_t = (in_gi.h_idx == 0 ? p.pht : 0);
  auto pad_h_b = (in_gi.h_idx + in_gi.h_slice == p.ih ? p.phb : 0);
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = p.has_bias ? module::isSign(getBias()) : 0;
    auto out_sign = module::isSign(getOutput());
    if (p.is_dw) {
      BM1684::instance().dl_nodechip_pooling_fix8b_forward_local(
          in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr, bottom_dim,
          top_dim, p.kh, p.kw, pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw,
          p.ins_h, p.ins_w,
          2, // is depthwise conv
          0, shift, p.has_bias,
          1, // shift type, useless param, but must be set...
          in_sign, filter_sign, bias_sign, out_sign, p.do_relu,
          BM1684::instance().bdc_node);
    } else {
      BM1684::instance().dl_nodechip_conv_forward_local_fix8b(
          in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
          gi.buffer_addr, bottom_dim, top_dim, p.groups, p.kh, p.kw, p.dh, p.dw,
          pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw, p.has_bias, 0, p.do_relu,
          p.relu_limit, /*unused_ht*/ 0, 0, 0, 0, /* insert h*/ p.ins_h,
          p.ins_w, shift, in_sign, filter_sign, bias_sign, true, /*mulshift*/ 0,
          0, 0, 0, BM1684::instance().bdc_node);
    }
  } else {
    BM1684::instance().dl_nodechip_conv_forward_local(
        in_gi.out_addr, f_gi.out_addr, b_gi.out_addr, gi.out_addr,
        gi.buffer_addr, bottom_dim, top_dim, p.groups, p.kh, p.kw, p.dh, p.dw,
        pad_h_t, pad_h_b, p.pwl, p.pwr, p.sh, p.sw, p.has_bias ? 1 : 0,
        /* result_add*/ 0, p.do_relu ? 1 : 0, p.relu_limit, 0, 0, 0, 0,
        BM1684::instance().bdc_node);
  }
}
