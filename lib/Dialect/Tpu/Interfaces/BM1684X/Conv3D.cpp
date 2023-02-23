//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
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

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape) {
  int32_t oc = shape[0];
  int32_t ic = shape[1];
  int32_t kd = shape[2];
  int32_t kh = shape[3];
  int32_t kw = shape[4];
  auto type_bytes = sizeof(T);
  int32_t IC_PARALLEL = BM168x::ic_num(type_bytes);
  auto kernel_hw = kh * kw;
  int32_t new_ic = ceiling_func(ic * kd, IC_PARALLEL);
  int32_t new_hw = kernel_hw * IC_PARALLEL;
  auto filter_new = std::make_shared<std::vector<T>>(oc * new_ic * new_hw, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic * kd)
            break;
          int orig_offset = oc_idx * ic * kd * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter->at(orig_offset);
        }
      }
    }
  }
  filter = filter_new;
  shape = {1, oc, new_ic, kh * kw, IC_PARALLEL};
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, int8_t>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();

  auto attr = op.parseParam();
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_type = module::getStorageType(op.getFilter());
  auto data_type = BM168x::getDataType(op.getFilter());
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  int64_t filter_shape[5];
  // (oc, ic, kd, kh, kw) -> (kd, oc, CEIL(ic,IC_PARALLEL), kh*kw, IC_PARALLEL)
  // int8/uint8 local layer shape, only change shape info for layer group
  filter_shape[0] = attr.kd;
  filter_shape[1] = attr.oc;
  filter_shape[2] = ceiling_func((attr.ic / attr.groups), IC_PARALLEL);
  filter_shape[3] = attr.kh * attr.kw;
  filter_shape[4] = IC_PARALLEL;
  auto new_type = RankedTensorType::get(filter_shape, filter_type);
  op.getFilter().setType(new_type);

  // bias op
  if (attr.has_bias) {
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type =
        RankedTensorType::get(bias_shape, module::getStorageType(op.getBias()));
    op.getBias().setType(new_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::Conv3DOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();
  if (attr.is_dw || attr.groups > 1) {
    llvm_unreachable("depthwise should support !!");
  }
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_type = module::getStorageType(op.getFilter());
  auto data_type = BM168x::getDataType(op.getFilter());
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t IC_PARALLEL = BM168x::ic_num(fmt_bytes);
  int64_t filter_shape[5];
  if (filter_type.isF16() || filter_type.isBF16()) {
    // (oc, ic, kd, kh, kw) -> (kd, oc, CEIL(ic,IC_PARALLEL), kh*kw, IC_PARALLEL)
    // f16/bf16 local layer shape, only change shape info for layer group
    filter_shape[0] = attr.kd;
    filter_shape[1] = attr.oc;
    filter_shape[2] = ceiling_func((attr.ic / attr.groups), IC_PARALLEL);
    filter_shape[3] = attr.kh * attr.kw;
    filter_shape[4] = IC_PARALLEL;
    auto new_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto bias_type = module::getStorageType(op.getBias());
    auto new_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, BFloat16Type>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, Float16Type>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, Float32Type>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();
  auto out_type = module::getStorageType(op.getOutput());
  // filter reorder
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  int64_t filter_shape[5];
  if (out_type.isF32()) {
    // (oc, ic, kd, kh, kw) -> (kd, oc,ic, kh, kw)
    // f32 local layer shape, only change shape info for layer group
    filter_shape[0] = attr.kd;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kh;
    filter_shape[4] = attr.kw;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.getFilter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.getBias().setType(new_type);
  }
  return success();
}

void tpu::Conv3DOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  conv3d_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.input_global_addr = module::getAddress(getInput());
  spec.weight_global_addr = module::getAddress(getFilter());
  spec.output_global_addr = module::getAddress(getOutput());
  if (attr.has_bias) {
    spec.has_bias = 1;
    spec.bias_global_addr = module::getAddress(getBias());
    spec.bias_dtype = BM168x::getDataType(getBias());
  }
  auto shape = module::getShape(getInput());
  for (size_t i = 0; i < shape.size(); ++i) {
    spec.input_shape[i] = shape[i];
  }
  spec.groups = attr.groups;
  spec.output_c = attr.oc;
  spec.kernel[0] = attr.kd;
  spec.kernel[1] = attr.kh;
  spec.kernel[2] = attr.kw;
  spec.stride[0] = attr.sd;
  spec.stride[1] = attr.sh;
  spec.stride[2] = attr.sw;
  spec.dilation[0] = attr.dd;
  spec.dilation[1] = attr.dh;
  spec.dilation[2] = attr.dw;
  spec.pad[0] = attr.pdf;
  spec.pad[1] = attr.pdb;
  spec.pad[2] = attr.pht;
  spec.pad[3] = attr.phb;
  spec.pad[4] = attr.pwl;
  spec.pad[5] = attr.pwr;
  spec.input_dtype = BM168x::getDataType(getInput());
  spec.weight_dtype = BM168x::getDataType(getFilter());
  spec.output_dtype = BM168x::getDataType(getOutput());
  spec.do_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto out_etype = module::getStorageType(getOutput());
    spec.do_relu = out_etype.isUnsignedInteger();
    auto in_qtype = module::getUniformQuantizedType(getInput());
    spec.kzp_is_const = true;
    spec.kzp_val = attr.kernel_zp;
    spec.kzp_dtype = spec.weight_dtype;
    spec.pad_is_const = true;
    spec.pad_val = in_qtype.getZeroPoint();
  }
  BM168x::call_global_func("backend_api_conv3d_global", &spec, sizeof(spec));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv3DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice,
    group_type_t group_type) {
  auto attr = parseParam();
  int64_t sz = 0;
  int64_t npu_num = BM168x::NPU_NUM;

  int32_t oc_per_npu = 0;
  for (int32_t i = 0; i < attr.groups; i++) {
    oc_per_npu =
        std::max((int64_t)oc_per_npu,
                 ceiling_func(i * attr.groups % npu_num + attr.oc / attr.groups,
                              npu_num));
  }
  auto in_type = getInput().getType();
  auto out_type = getOutput().getType();
  // output start npu id must be same with weight start npu id
  if ((in_type.isF16() || in_type.isBF16()) && !out_type.isF32() &&
      attr.kd > 1) {
    sz += (oc_per_npu *
           align_up(out_hslice * attr.ow, BM168x::eu_num(sizeof(float))) *
           sizeof(float));
  }

  // input must start from npu 0
  if ((in_type.isF16() || in_type.isBF16()) && attr.groups > 1) {
    sz += ceiling_func((int64_t)attr.ic / attr.groups, npu_num) *
          align_up(in_hslice * attr.iw, BM168x::eu_num(sizeof(int16_t))) *
          sizeof(int16_t);
  }
  if ((in_type.isInteger(8)) && attr.groups > 1) {
    sz += ceiling_func((int64_t)attr.ic / attr.groups, npu_num) *
          align_up(in_hslice * attr.iw, BM168x::eu_num(sizeof(int8_t))) *
          sizeof(int8_t);
  }
  return sz;
}

void tpu::Conv3DOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);

  conv3d_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.input_local_addr = in_gi.out_addr;
  spec.weight_local_addr =
      LocalGenInterface::getGroupInfo(getFilter()).out_addr;
  if (attr.has_bias) {
    spec.has_bias = true;
    spec.bias_local_addr = LocalGenInterface::getGroupInfo(getBias()).out_addr;
    spec.bias_dtype = BM168x::getDataType(getBias());
  }
  spec.buffer_local_addr = gi.buffer_addr;
  spec.output_local_addr = gi.out_addr;
  spec.input_shape[0] = attr.id;
  spec.input_shape[1] = sec_info.n_slice;
  spec.input_shape[2] = attr.ic;
  spec.input_shape[3] = sec_info.h_slice;
  spec.input_shape[4] = attr.iw;
  spec.groups = attr.groups;
  spec.output_c = attr.oc;
  spec.kernel[0] = attr.kd;
  spec.kernel[1] = attr.kh;
  spec.kernel[2] = attr.kw;
  spec.stride[0] = attr.sd;
  spec.stride[1] = attr.sh;
  spec.stride[2] = attr.sw;
  spec.dilation[0] = attr.dd;
  spec.dilation[1] = attr.dh;
  spec.dilation[2] = attr.dw;
  spec.pad[0] = attr.pdf;
  spec.pad[1] = attr.pdb;
  spec.pad[2] = sec_info.h_idx == 0 ? attr.pht : 0;
  spec.pad[3] = sec_info.h_idx + sec_info.h_slice >= attr.ih ? attr.phb : 0;
  spec.pad[4] = attr.pwl;
  spec.pad[5] = attr.pwr;
  spec.input_dtype = BM168x::getDataType(getInput());
  spec.weight_dtype = BM168x::getDataType(getFilter());
  spec.output_dtype = BM168x::getDataType(getOutput());
  spec.do_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto out_etype = module::getStorageType(getOutput());
    spec.do_relu = out_etype.isUnsignedInteger(8);
    auto in_qtype = module::getUniformQuantizedType(getInput());
    spec.kzp_is_const = true;
    spec.kzp_val = attr.kernel_zp;
    spec.kzp_dtype = spec.weight_dtype;
    spec.pad_is_const = true;
    spec.pad_val = in_qtype.getZeroPoint();
  }
  BM168x::call_local_func("backend_api_conv3d_local", &spec,
                          sizeof(conv3d_local_spec_t));
}

// dynamic codegen
int64_t tpu::Conv3DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(dyn_conv3d_local_param_t);
  auto attr = parseParam();
  auto gi = getGroupInfo(0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);

  dyn_conv3d_local_param_t param = {0};
  if (attr.has_bias) {
    param.spec.common.has_bias = true;
    param.spec.common.bias_dtype = BM168x::getDataType(getBias());
  }
  param.spec.buffer_local_addr = gi.buffer_addr;
  param.spec.common.groups = attr.groups;
  param.spec.common.output_c = attr.oc;
  param.spec.common.kernel[0] = attr.kd;
  param.spec.common.kernel[1] = attr.kh;
  param.spec.common.kernel[2] = attr.kw;
  param.spec.common.stride[0] = attr.sd;
  param.spec.common.stride[1] = attr.sh;
  param.spec.common.stride[2] = attr.sw;
  param.spec.common.dilation[0] = attr.dd;
  param.spec.common.dilation[1] = attr.dh;
  param.spec.common.dilation[2] = attr.dw;
  param.spec.common.pad[0] = attr.pdf;
  param.spec.common.pad[1] = attr.pdb;
  param.spec.common.pad[2] = attr.pht;
  param.spec.common.pad[3] = attr.phb;
  param.spec.common.pad[4] = attr.pwl;
  param.spec.common.pad[5] = attr.pwr;
  param.spec.common.input_dtype = BM168x::getDataType(getInput());
  param.spec.common.weight_dtype = BM168x::getDataType(getFilter());
  param.spec.common.output_dtype = BM168x::getDataType(getOutput());
  param.spec.common.do_relu = attr.do_relu;
  param.spec.common.relu_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto out_etype = module::getStorageType(getOutput());
    param.spec.common.do_relu = out_etype.isUnsignedInteger(8);
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.spec.common.kzp_is_const = true;
    param.spec.common.kzp_val = attr.kernel_zp;
    param.spec.common.kzp_dtype = param.spec.common.weight_dtype;
    param.spec.common.pad_is_const = true;
    param.spec.common.pad_val = in_qtype.getZeroPoint();
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Conv3DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(dyn_conv3d_global_param_t);
  auto attr = parseParam();
  dyn_conv3d_global_param_t param = {0};
  if (attr.has_bias) {
    param.spec.common.has_bias = 1;
    param.spec.common.bias_dtype = BM168x::getDataType(getBias());
  }

  param.spec.common.groups = attr.groups;
  param.spec.common.output_c = attr.oc;
  param.spec.common.kernel[0] = attr.kd;
  param.spec.common.kernel[1] = attr.kh;
  param.spec.common.kernel[2] = attr.kw;
  param.spec.common.stride[0] = attr.sd;
  param.spec.common.stride[1] = attr.sh;
  param.spec.common.stride[2] = attr.sw;
  param.spec.common.dilation[0] = attr.dd;
  param.spec.common.dilation[1] = attr.dh;
  param.spec.common.dilation[2] = attr.dw;
  param.spec.common.pad[0] = attr.pdf;
  param.spec.common.pad[1] = attr.pdb;
  param.spec.common.pad[2] = attr.pht;
  param.spec.common.pad[3] = attr.phb;
  param.spec.common.pad[4] = attr.pwl;
  param.spec.common.pad[5] = attr.pwr;
  param.spec.common.input_dtype = BM168x::getDataType(getInput());
  param.spec.common.weight_dtype = BM168x::getDataType(getFilter());
  param.spec.common.output_dtype = BM168x::getDataType(getOutput());
  param.spec.common.do_relu = attr.do_relu;
  param.spec.common.relu_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto out_etype = module::getStorageType(getOutput());
    param.spec.common.do_relu = out_etype.isUnsignedInteger();
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.spec.common.kzp_is_const = true;
    param.spec.common.kzp_val = attr.kernel_zp;
    param.spec.common.kzp_dtype = param.spec.common.weight_dtype;
    param.spec.common.pad_is_const = true;
    param.spec.common.pad_val = in_qtype.getZeroPoint();
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::Conv3DOp::get_fw_type_bm1684x() {
  return FW_BMNET_CONV3D;
}
