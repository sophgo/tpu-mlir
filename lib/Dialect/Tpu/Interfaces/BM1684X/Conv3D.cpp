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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
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
  if (!Module::getStorageType(op.filter()).isInteger(8))
    return failure();

  conv_attr_t attr = {0};
  op.parseParam(&attr);
  // filter
  auto filterOp = op.filter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.groups, attr.kd,
                                       attr.kh, attr.kw};
  // (oc, ic, kd, kh, kw) -> (oc, (ic*kt)/64, kh, kw, 64)
  filter_reorder(filter_i8, filter_shape);

  OpBuilder builder(getContext());
  auto elem_type = Module::getStorageType(op.filter());
  auto filter_type = RankedTensorType::get(filter_shape, elem_type);
  auto new_filter =
      top::WeightOp::create(op, "reordered", *filter_i8, filter_type);
  op->setOperand(1, new_filter);

  // bias op
  if (attr.has_bias) {
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type =
        RankedTensorType::get(bias_shape, Module::getStorageType(op.bias()));
    op.bias().setType(new_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::Conv3DOp op,
                                          PatternRewriter &rewriter) {
  conv_attr_t attr = {0};
  op.parseParam(&attr);
  auto filterOp = op.filter().getDefiningOp<top::WeightOp>();
  if (attr.is_dw || attr.groups > 1) {
    llvm_unreachable("depthwise should support !!");
  }
  auto filter_u16 = filterOp.read<uint16_t>();
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic, attr.kd, attr.kh,
                                       attr.kw};
  filter_reorder(filter_u16, filter_shape);

  auto filter_type = Module::getStorageType(op.filter());
  auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
  auto newFilterOp =
      top::WeightOp::create(op, "reordered", *filter_u16, new_filter_type);
  op->setOperand(1, newFilterOp);

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.bias().getDefiningOp<top::WeightOp>();
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type =
        RankedTensorType::get(bias_shape, Module::getStorageType(op.bias()));
    op.bias().setType(new_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, BFloat16Type>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!Module::getStorageType(op.filter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, Float16Type>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!Module::getStorageType(op.filter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::Conv3DOp, Float32Type>::matchAndRewrite(
    tpu::Conv3DOp op, PatternRewriter &rewriter) const {
  if (!Module::getStorageType(op.filter()).isF32())
    return failure();

  conv_attr_t attr = {0};
  op.parseParam(&attr);
  auto out_type = Module::getStorageType(op.output());
  // filter reorder
  auto filterOp = op.filter().getDefiningOp<top::WeightOp>();
  int64_t filter_shape[5];
  if (out_type.isF32()) {
    filter_shape[0] = 1;
    filter_shape[1] = attr.oc;
    filter_shape[2] = attr.ic / attr.groups;
    filter_shape[3] = attr.kd * attr.kh * attr.kw;
    filter_shape[4] = 1;
    auto new_type = RankedTensorType::get(filter_shape, out_type);
    op.filter().setType(new_type);
  } else {
    op.dump();
    llvm_unreachable("op type not support");
  }

  // bias op
  if (attr.has_bias) {
    auto biasOp = op.bias().getDefiningOp<top::WeightOp>();
    llvm::SmallVector<int64_t> bias_shape = {1, attr.oc, 1, 1, 1};
    auto new_type = RankedTensorType::get(bias_shape, out_type);
    op.bias().setType(new_type);
  }
  return success();
}

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t input_global_addr;
  uint64_t weight_global_addr;
  uint64_t bias_global_addr;
  uint64_t output_global_addr;
  int32_t input_shape[5]; // (n, ic, it, ih, iw)
  int32_t groups;
  int32_t output_c;
  int32_t kernel[3];
  int32_t stride[3];
  int32_t dilation[3];
  int32_t pad[6];
  int32_t has_bias;
  int32_t input_dtype;
  int32_t weight_dtype;
  int32_t bias_dtype;
  int32_t output_dtype;
  int32_t do_relu;
  float relu_limit;
  uint64_t kzp_global_addr;
  uint64_t pad_global_addr;
  bool kzp_is_const;
  bool pad_is_const;
  int32_t kzp_val;
  int32_t pad_val;
  int32_t kzp_dtype;
} conv3d_global_spec_t;

typedef struct conv_local_param {
  uint32_t input_local_addr;
  uint32_t weight_local_addr;
  uint32_t bias_local_addr;
  uint32_t buffer_local_addr;
  uint32_t output_local_addr;
  int32_t input_shape[5]; // (id, n, ic, ih, iw)
  int32_t groups;
  int32_t output_c;
  int32_t kernel[3];   // (kd, kh, kw)
  int32_t stride[3];   // (sd, sh, sw)
  int32_t dilation[3]; // (dd, dh, dw)
  int32_t pad[6];      // (df, db, ht, hb, wl, wr)
  int32_t has_bias;
  int32_t input_dtype;
  int32_t weight_dtype;
  int32_t bias_dtype;
  int32_t output_dtype;
  int32_t do_relu;
  float relu_limit;
  uint32_t kzp_local_addr;
  uint32_t pad_local_addr;
  bool kzp_is_const;
  bool pad_is_const;
  int32_t kzp_val;
  int32_t pad_val;
  int32_t kzp_dtype;
} conv3d_local_spec_t;

#ifdef __cplusplus
}
#endif

void tpu::Conv3DOp::codegen_global_bm1684x() {
  conv_attr_t attr = {0};
  parseParam(&attr);
  conv3d_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.input_global_addr = Module::getAddress(input());
  spec.weight_global_addr = Module::getAddress(filter());
  spec.output_global_addr = Module::getAddress(output());
  if (attr.has_bias) {
    spec.has_bias = 1;
    spec.bias_global_addr = Module::getAddress(bias());
    spec.bias_dtype = BM168x::getDataType(bias());
  }
  auto shape = Module::getShape(input());
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
  spec.input_dtype = BM168x::getDataType(input());
  spec.weight_dtype = BM168x::getDataType(filter());
  spec.output_dtype = BM168x::getDataType(output());
  spec.do_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  if (Quant::isUniformQuantized(input())) {
    auto out_etype = Module::getStorageType(output());
    spec.do_relu = out_etype.isUnsignedInteger(8);
    auto in_qtype = Quant::getUniformQuantizedType(input());
    spec.kzp_is_const = true;
    spec.kzp_val = attr.kernel_zp;
    spec.kzp_dtype = spec.weight_dtype;
    spec.pad_is_const = true;
    spec.pad_val = in_qtype.getZeroPoint();
  }
  auto op = getOperation();
  BM168x::call_global_func("backend_api_conv3d_global", &spec, sizeof(spec));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::Conv3DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  conv_attr_t attr = {0};
  parseParam(&attr);
  int64_t sz = 0;
  auto chip = Module::getChip(getOperation());
  int64_t npu_num = BM168x::NPU_NUM;

  int32_t oc_per_npu = 0;
  for (int32_t i = 0; i < attr.groups; i++) {
    oc_per_npu =
        std::max((int64_t)oc_per_npu,
                 ceiling_func(i * attr.groups % npu_num + attr.oc / attr.groups,
                              npu_num));
  }
  auto in_type = input().getType();
  auto out_type = output().getType();
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

void tpu::Conv3DOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  conv_attr_t attr = {0};
  parseParam(&attr);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  conv3d_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.input_local_addr = in_gi.out_addr;
  spec.weight_local_addr = LocalGenInterface::getGroupInfo(filter()).out_addr;
  if (attr.has_bias) {
    spec.has_bias = true;
    spec.bias_local_addr = LocalGenInterface::getGroupInfo(bias()).out_addr;
    spec.bias_dtype = BM168x::getDataType(bias());
  }
  spec.buffer_local_addr = gi.buffer_addr;
  spec.output_local_addr = gi.out_addr;
  spec.input_shape[0] = attr.id;
  spec.input_shape[1] = in_gi.n_slice;
  spec.input_shape[2] = attr.ic;
  spec.input_shape[3] = in_gi.h_slice;
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
  spec.pad[2] = in_gi.h_idx == 0 ? attr.pht : 0;
  spec.pad[3] = in_gi.h_idx + in_gi.h_slice >= attr.ih ? attr.phb : 0;
  spec.pad[4] = attr.pwl;
  spec.pad[5] = attr.pwr;
  spec.input_dtype = BM168x::getDataType(input());
  spec.weight_dtype = BM168x::getDataType(filter());
  spec.output_dtype = BM168x::getDataType(output());
  spec.do_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  if (Quant::isUniformQuantized(input())) {
    auto out_etype = Module::getStorageType(output());
    spec.do_relu = out_etype.isUnsignedInteger(8);
    auto in_qtype = Quant::getUniformQuantizedType(input());
    spec.kzp_is_const = true;
    spec.kzp_val = attr.kernel_zp;
    spec.kzp_dtype = spec.weight_dtype;
    spec.pad_is_const = true;
    spec.pad_val = in_qtype.getZeroPoint();
  }
  BM168x::call_local_func("backend_api_conv3d_local", &spec,
                          sizeof(conv3d_local_spec_t));
}
