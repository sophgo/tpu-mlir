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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;
// ======================================
// WeightReorderInterface
// ======================================

// refer to net_compiler: bool BM1684XCoeffArranger::ConvWeightArr(GraphEdge*
// edge)
template <>
LogicalResult WeightReorder<tpu::Conv1DOp, int8_t>::matchAndRewrite(
    tpu::Conv1DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8) ||
      op.getCoeffMerged())
    return failure();

  auto attr = op.parseParam();

  bool merge = true;
  auto out_stype = module::getStorageType(op.getOutput());
  if (out_stype.isInteger(32)) {
    merge = false;
  }

  // filter
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kh};
  int IC_PARALLEL = BM168x::ic_num(1);
  int use_3ic_optimize = 0;
  if (attr.ic * attr.kh <= IC_PARALLEL && attr.kh > 1) {
    use_3ic_optimize = 1; // merge kh to ic
  } else {
    use_3ic_optimize = 0;
  }
  if (use_3ic_optimize) {
    // Now only support broadcast using BDC when it is a local layer.
    use_3ic_optimize |= 0x10;
  }

  if (attr.is_dw == false) {
    tpu::reshape_coeff_for_3ic(filter_i8, filter_shape, use_3ic_optimize);
  } else {
    filter_shape = {1, attr.oc, attr.kh};
  }
  op->setAttr("use_3ic_optimize", rewriter.getI64IntegerAttr(use_3ic_optimize));
  if (merge == false) {
    auto stype = module::getStorageType(op.getFilter());
    auto new_type = RankedTensorType::get(filter_shape, stype);
    auto new_op =
        top::WeightOp::create(op, "filter_reorderd", *filter_i8, new_type);
    op->setOperand(1, new_op);
    if (attr.has_bias) {
      auto elem_type = module::getStorageType(op.getBias());
      auto bias_type = RankedTensorType::get({1, attr.oc, 1}, elem_type);
      op.getBias().setType(bias_type);
    }
    return success();
  }
  tpu::reshape_coeff_for_broadcast_channel(filter_i8, filter_shape, false);
  int64_t new_oc = filter_shape[1];
  int64_t filter_w_bytes = filter_shape[2] * sizeof(int8_t);

  // bias
  i32_array_t bias_new;
  std::vector<int64_t> bias_shape = {1, attr.oc, 1};
  int64_t bias_w_bytes = 0;
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    bias_new = biasOp.read<int32_t>();
    tpu::reshape_coeff_for_broadcast_channel(bias_new, bias_shape, false);
    assert(new_oc == bias_shape[1]);
    bias_w_bytes = bias_shape[2] * sizeof(int32_t);
  }

  // requant
  auto qtype = module::getUniformQuantizedType(op.getOutput());
  std::vector<int64_t> quant_shape = {1, attr.oc, 3};
  auto quant_data = std::make_shared<std::vector<int32_t>>(attr.oc * 3, 0);
  auto m_data = module::getI64Array(op.getMultiplier(), attr.oc, 1);
  auto r_data = module::getI64Array(op.getRshift(), attr.oc, 0);
  for (int i = 0; i < attr.oc; i++) {
    quant_data->at(i * 3) = m_data->at(i);
    quant_data->at(i * 3 + 1) = -r_data->at(i);
    quant_data->at(i * 3 + 2) = qtype.getZeroPoint();
  }
  tpu::reshape_coeff_for_broadcast_channel(quant_data, quant_shape, true);
  assert(new_oc == quant_shape[1]);
  int64_t quant_w_bytes = quant_shape[2] * sizeof(int32_t);

  // merge
  int64_t quant_offset = 0, bias_offset = 0, filter_offset = 0;
  int64_t filter_align = attr.is_dw ? 1 : BM168x::EU_BYTES;

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
  std::vector<int64_t> coeff_shape = {1, new_oc, merge_w};
  for (int i = 0; i < new_oc; i++) {
    auto coeff_ptr = new_coeff->data() + i * merge_w;
    auto quant_ptr = quant_data->data() + i * quant_shape[2];
    auto bias_ptr =
        attr.has_bias ? (bias_new->data() + i * bias_shape[2]) : nullptr;
    auto filter_ptr = filter_i8->data() + i * filter_shape[2];
    // copy quant
    memcpy(coeff_ptr + quant_offset, quant_ptr, quant_w_bytes);
    if (attr.has_bias) {
      memcpy(coeff_ptr + bias_offset, bias_ptr, bias_w_bytes);
    }
    memcpy(coeff_ptr + filter_offset, filter_ptr, filter_w_bytes);
  }

  auto elem_type = module::getStorageType(op.getFilter());
  auto coeff_type = RankedTensorType::get(coeff_shape, elem_type);
  auto coeff_op = top::WeightOp::create(op, "merge", *new_coeff, coeff_type);
  op->removeAttr("rshift");
  op->removeAttr("multiplier");
  op->setAttr("coeff_merged", rewriter.getBoolAttr(true));
  op->setOperand(1, coeff_op);
  auto none = module::getNoneOp(op);
  op->setOperand(2, none.getResult());
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::Conv1DOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  if (attr.is_dw || attr.groups > 1) {
    llvm_unreachable("depthwise should support !!");
  }
  auto filter_u16 = filterOp.read<uint16_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic, attr.kh};
  tpu::filter_reorder(filter_u16, filter_shape);
  tpu::reshape_coeff_for_broadcast_channel(filter_u16, filter_shape);

  auto filter_type = module::getStorageType(op.getFilter());
  auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
  auto newFilterOp =
      top::WeightOp::create(op, "_reordered", *filter_u16, new_filter_type);
  op->setOperand(1, newFilterOp);

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    int64_t bias_shape[3] = {1, attr.oc, 1};
    auto new_type =
        RankedTensorType::get(bias_shape, module::getStorageType(op.getBias()));
    op.getBias().setType(new_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv1DOp, BFloat16Type>::matchAndRewrite(
    tpu::Conv1DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv1DOp, Float16Type>::matchAndRewrite(
    tpu::Conv1DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv1DOp, Float32Type>::matchAndRewrite(
    tpu::Conv1DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  int64_t filter_shape[3];
  if (out_type.isF32()) {
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups * attr.kh;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    int64_t bias_shape[3] = {1, attr.oc, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}

// ======================================
// GlobalGenInterface
// ======================================

void tpu::Conv1DOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  (*input_spec)[0].dims = 4;
  (*input_spec)[0].shape[3] = 1;
  (*output_spec)[0].dims = 4;
  (*output_spec)[0].shape[3] = 1;
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
  common.round_mode = ROUND_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    if (getCoeffMerged()) {
      spec.merge_coeff = 2;
      auto out_etype = module::getStorageType(getOutput());
      if (out_etype.isUnsignedInteger()) {
        common.if_relu = true;
      }
    }
    common.is_asym = true;
    common.ipad_value = in_qtype.getZeroPoint();
    common.use_3ic_optimize = getUse_3icOptimize();
  }
  BM168x::call_global_func("backend_api_conv_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv1DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  auto p = parseParam();
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
  int use_3ic = (getUse_3icOptimize() & 0x3);
  if (use_3ic == 1) { // merge kh to ic
    sz += align_up(out_hslice * in_wslice, eu_num) * in_nslice * in_type_len;
    sz += 64 * 2;
    sz += p.kh * in_type_len;
  } else if (use_3ic == 2 | use_3ic == 3) {
    llvm_unreachable("wrong 3ic for conv1d");
  }
  return sz;
}

void tpu::Conv1DOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step, d_step, w_step);

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
  common.round_mode = ROUND_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    if (getCoeffMerged()) {
      p.spec.merge_coeff = 2;
      p.spec.with_requant = 1;
      auto out_etype = module::getStorageType(getOutput());
      if (out_etype.isUnsignedInteger()) {
        common.if_relu = true;
      }
    }
    common.is_asym = true;
    common.ipad_value = in_qtype.getZeroPoint();
    common.use_3ic_optimize = getUse_3icOptimize();
  }

  BM168x::call_local_func("backend_api_conv_local", &p, sizeof(p), &sec_info,
                          input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::Conv1DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(conv_local_param_t);
  auto gi = getGroupInfo(0, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);
  auto attr = parseParam();
  conv_local_param_t p = {0};
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
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    if (getCoeffMerged()) {
      p.spec.merge_coeff = 2;
      p.spec.with_requant = 1;
      auto out_etype = module::getStorageType(getOutput());
      if (out_etype.isUnsignedInteger()) {
        common.if_relu = true;
      }
    }
    common.is_asym = true;
    common.ipad_value = in_qtype.getZeroPoint();
    common.use_3ic_optimize = getUse_3icOptimize();
  }
  return BM168x::dynamic_spec_to_buffer(buffer, p);
 }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Conv1DOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  //now nodechip justsupport ndims = 4, Todo
  assert((*input_spec)[0].dims == 4);
  if (!buffer) return sizeof(conv_global_spec_t);
  conv_global_spec_t spec = {0};
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
  common.round_mode = ROUND_UP;
  common.has_bias = attr.has_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.kzp_is_const = true;
  common.kzp_value = attr.kernel_zp;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    if (getCoeffMerged()) {
      spec.merge_coeff = 2;
      auto out_etype = module::getStorageType(getOutput());
      if (out_etype.isUnsignedInteger()) {
        common.if_relu = true;
      }
    }
    common.is_asym = true;
    common.ipad_value = in_qtype.getZeroPoint();
    common.use_3ic_optimize = getUse_3icOptimize();
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::Conv1DOp::get_fw_type_bm1684x() {
  return FW_BMNET_CONV;
}
