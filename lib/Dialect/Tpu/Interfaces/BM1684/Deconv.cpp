//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "mlir/Support/LogicalResult.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684;

void deconv_weight_transform(int ic, int oc, int kh, int kw,
                           const void *weight_orig, const void *weight_trans,
                           int type_bytes) {
  int trans_offset;
  int hw = kw* kh;
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < ic; ic_idx++) {
      for(int k_idx=0; k_idx< hw; k_idx++) {
        int orig_offset = ic_idx *oc * hw  + k_idx + oc_idx * hw;
        orig_offset = oc_idx*ic*hw  + ic_idx*hw  + k_idx;
        switch (type_bytes) {
        case 4:
          trans_offset = oc_idx*align_up(ic,2)*hw + ic_idx/2*hw*2 + k_idx*2 + ic_idx%2;
          *((float *)weight_trans + trans_offset) =
              *((float *)weight_orig + orig_offset);
          break;
        default:
          llvm_unreachable("wrong conv weight data type");
        }
      }
    }
  }
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, int8_t>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  auto attr = op.parseParam();
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_int8 = filterOp.read<int8_t>();
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  int new_size = attr.oc * attr.g * (align_up(attr.ic, 4l)) * attr.kh * attr.kw;
  std::vector<int64_t> new_shape = {1, attr.oc* attr.g, attr.kh * attr.kw * align_up(attr.ic, 4l), 1};
  auto new_type = RankedTensorType::get(new_shape, filter_type.getElementType());
  auto filter_new = std::make_shared<std::vector<int8_t>>(new_size, 0);
  for (int oc_idx = 0; oc_idx < attr.oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < attr.ic; ic_idx++) {
      for (int k_idx = 0; k_idx < attr.kh * attr.kw; k_idx++) {
        int orig_offset = oc_idx*attr.ic*attr.kh * attr.kw + ic_idx*attr.kh * attr.kw + k_idx;
        int trans_offset = oc_idx * align_up(attr.ic,4l)*attr.kw*attr.kh + ic_idx/4*attr.kh * attr.kw *4 + k_idx*4 + ic_idx%4;
        filter_new->at(trans_offset) = filter_int8->at(orig_offset);
      }
    }
  }
   auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                              "reorderd", *filter_new, new_type);
  op->setOperand(1, new_filter);
  return success();
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, Float32Type>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();
  auto attr = op.parseParam();
  auto type_bytes = 4;
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto weight_data = filterOp.read_as_byte();
  if (attr.is_dw == false) {
    std::vector<int64_t> new_shape = {1, attr.oc * attr.g, align_up(attr.ic, 2l)/2,
                        attr.kh * attr.kw *2};
    int new_count =
        align_up(attr.ic, 2l) * attr.oc * attr.g * attr.kh * attr.kw;
    auto filter_new = std::make_shared<std::vector<float>>(new_count, 0);
    deconv_weight_transform( attr.ic, attr.oc * attr.g,attr.kh, attr.kw,
                          weight_data->data(), filter_new->data(), type_bytes);
    auto new_type = RankedTensorType::get(new_shape, out_type);
    auto new_filter = top::WeightOp::create(op.getFilter().getDefiningOp(),
                                            "reorderd", *filter_new, new_type);
    op->setOperand(1, new_filter);

  } else {
    int64_t filter_shape[4];
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc * attr.g;
    filter_shape[2] = align_up(attr.ic, 2l)/2 ;
    filter_shape[3] = attr.kh * attr.kw *2;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);


  }
  // bias op
  if (attr.with_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
}

void tpu::DeconvOp::codegen_global_bm1684() {
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
      auto bias_sign = attr.with_bias ? module::isSign(getBias()) : 0;
      BM1684::instance().dl_nodechip_deconv_fix8b_forward_parallel(
        in_addr, out_addr, filter_addr, bias_addr,
              attr.n, attr.ic, attr.ih, attr.iw,
              attr.oc, attr.g, attr.kh, attr.kw, attr.dh, attr.dw,
              attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after,
              attr.sh, attr.sw,
              attr.output_pad_h, attr.output_pad_w,
              attr.with_bias ? 1 : 0, attr.do_relu ? 1 : 0,
              shift, 1, in_sign, filter_sign, bias_sign,
              (CMD_ID_NODE *)BM1684::instance().cmdid_node
              );
  } else {
  if (attr.is_dw) {
      BM1684::instance().dl_nodechip_depthwise_forward_parallel(
          in_addr, out_addr, filter_addr, bias_addr, attr.n, attr.ic, attr.ih,
          attr.iw, attr.kh, attr.kw, attr.pad_h, attr.pad_h_after, attr.pad_w, attr.pad_w_after,
          attr.sh, attr.sw, attr.dh, attr.dw, attr.with_bias ? 1 : 0,
          attr.do_relu ? 1 : 0, attr.relu_limit,
          (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      BM1684::instance().dl_nodechip_deconv_forward_parallel_with_data_split_v2(
            in_addr,
            out_addr,
            filter_addr,
            bias_addr,
            attr.n,
            attr.ic,
            attr.ih,
            attr.iw,
            attr.g,
            attr.oc,
            attr.kh,
            attr.kw,
            attr.dh,
            attr.dw,
            attr.pad_h,
            attr.pad_h_after,
            attr.pad_w,
            attr.pad_w_after,
            attr.sh,
            attr.sw,
            attr.output_pad_h,attr.output_pad_w,
            attr.with_bias ? 1 : 0,
            0,
            attr.do_relu ? 1 : 0,
            1,
            1,
            (CMD_ID_NODE *)BM1684::instance().cmdid_node
            );}
  }
}

int64_t tpu::DeconvOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  // TODO for spicial situation
  return 0;
}

void tpu::DeconvOp::codegen_local_bm1684(int64_t n_step, int64_t h_step, local_sec_info_t &sec_info) {
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto f_gi = LocalGenInterface::getGroupInfo(getFilter());
  auto b_gi = LocalGenInterface::getGroupInfo(getBias());
  auto gi = getGroupInfo(n_step, h_step, 0, 0);
  auto p = parseParam();
  int bottom_dim[4] = {(int)in_gi.n_slice, (int)p.ic, (int)in_gi.h_slice,
                       (int)p.iw};
  int top_dim[4] = {(int)gi.n_slice, (int)p.oc, (int)gi.h_slice, (int)p.ow};
  int kh_ext = (p.kh - 1) * p.dh + 1;
  if (auto deconv_in_slice = DeconvSlice(gi.h_idx, gi.h_slice, p.sh, kh_ext,
                                         p.ih, p.pad_h)) {
    p.pad_h = deconv_in_slice.value()[0];
    p.pad_h_after = deconv_in_slice.value()[1];
  } else {
    p.pad_h = p.kh - p.pad_h - 1;
    p.pad_h_after = p.kh - p.pad_h_after - 1 + p.output_pad_h;
  }
  p.pad_w = p.kw - p.pad_w - 1;
  p.pad_w_after = p.kw - p.pad_w_after - 1 + p.output_pad_w;
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    auto shift = shift_v->at(0);
    auto in_sign = module::isSign(getInput());
    auto filter_sign = module::isSign(getFilter());
    auto bias_sign = p.with_bias ? module::isSign(getBias()) : 0;
    BM1684::instance().dl_nodechip_deconv_fix8b_forward_local(
      in_gi.out_addr,
      f_gi.out_addr,
      b_gi.out_addr,
      gi.out_addr,
      bottom_dim,
      top_dim,
      p.g,
      p.kh, p.kw, p.dh, p.dw,
      p.pad_h,
      p.pad_h_after,
      p.pad_w,
      p.pad_w_after,
      p.sh-1, p.sw-1,
      p.with_bias ? 1 : 0, p.do_relu ? 1 : 0, shift, in_sign, filter_sign,  bias_sign,
      (CMD_ID_NODE *)BM1684::instance().bdc_node);
  } else {
    BM1684::instance().dl_nodechip_deconv_forward_local(
      in_gi.out_addr,
      f_gi.out_addr,
      b_gi.out_addr,
      gi.out_addr,
      bottom_dim, top_dim,
      p.g,
      p.kh,
      p.kw,
      p.dh,
      p.dw,
      p.pad_h,
      p.pad_h_after,
      p.pad_w,
      p.pad_w_after,
      p.sh-1, p.sw-1,
      p.with_bias,
      0,
      p.do_relu ? 1 : 0,
      (CMD_ID_NODE *)BM1684::instance().bdc_node);
  }
}

uint32_t tpu::DeconvOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::DeconvOp::get_fw_type_bm1684() {
  return -1;
}

int32_t tpu::DeconvOp::dyn_codegen_local_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}