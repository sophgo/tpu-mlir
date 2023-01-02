//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

template <typename T>
static void filter_reorder(std::shared_ptr<std::vector<T>> &filter,
                           std::vector<int64_t> &shape) {
  int64_t oc, ic, kh, kw;
  module::getNCHW(shape, oc, ic, kh, kw);
  auto type_bytes = sizeof(T);
  int64_t IC_PARALLEL = BM168x::ic_num(type_bytes);
  auto kernel_hw = kh * kw;
  int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
  int64_t new_hw = kernel_hw * IC_PARALLEL;
  auto filter_new = std::make_shared<std::vector<T>>(oc * new_ic * new_hw, 0);
  for (int oc_idx = 0; oc_idx < oc; oc_idx++) {
    for (int ic_idx = 0; ic_idx < new_ic; ic_idx++) {
      for (int k_idx = 0; k_idx < kernel_hw; k_idx++) {
        for (int inner = 0; inner < IC_PARALLEL; inner++) {
          if (ic_idx * IC_PARALLEL + inner >= ic)
            break;
          int orig_offset = oc_idx * ic * kh * kw +
                            (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx;
          int trans_offset = oc_idx * new_ic * new_hw + ic_idx * new_hw +
                             k_idx * IC_PARALLEL + inner;
          filter_new->at(trans_offset) = filter->at(orig_offset);
        }
      }
    }
  }
  filter = filter_new;
  shape = {1, oc, 1, new_ic * new_hw};
}

// refer to net_compiler: bool BM1684XCoeffArranger::DeconvWeightArr(GraphEdge*
// edge)
template <>
LogicalResult WeightReorder<tpu::DeconvOp, int8_t>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isInteger(8))
    return failure();
  // assume that ic = input_channel / groups, oc = output_channel / groups
  // for original model, deconv kernel is {groups * ic, oc, kh, kw},
  // but kernel is arranged to {groups * oc, ic, kh, kw} when adding_layer
  // here we arrange kernel to {groups * oc, ceil(ic, IC_PARALLEL), kh * kw *
  // IC_PARALLEL}
  auto attr = op.parseParam();

  // filter op
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_i8 = filterOp.read<int8_t>();
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  if (attr.is_dw) {
    filter_shape = {1, attr.oc, attr.kh, attr.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_filter_type);
  } else {
    filter_reorder(filter_i8, filter_shape);
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    auto newFilterOp =
        top::WeightOp::create(op, "_reordered", *filter_i8, new_filter_type);
    op->setOperand(1, newFilterOp);
  }

  // bias op
  if (attr.with_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

LogicalResult weight_reorder_bf16_bm1684x(tpu::DeconvOp op,
                                          PatternRewriter &rewriter) {
  auto attr = op.parseParam();

  // filter op
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_u16 = filterOp.read<uint16_t>();
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {attr.oc, attr.ic / attr.g, attr.kh,
                                       attr.kw};
  if (attr.is_dw) {
    filter_shape = {1, attr.oc, attr.kh, attr.kw};
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    op.getFilter().setType(new_filter_type);
  } else {
    filter_reorder(filter_u16, filter_shape);
    auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
    auto newFilterOp =
        top::WeightOp::create(op, "_reordered", *filter_u16, new_filter_type);
    op->setOperand(1, newFilterOp);
  }

  // bias op
  if (attr.with_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, BFloat16Type>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isBF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, Float16Type>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF16())
    return failure();
  return weight_reorder_bf16_bm1684x(op, rewriter);
}

template <>
LogicalResult WeightReorder<tpu::DeconvOp, Float32Type>::matchAndRewrite(
    tpu::DeconvOp op, PatternRewriter &rewriter) const {
  if (!module::getStorageType(op.getFilter()).isF32())
    return failure();

  auto attr = op.parseParam();

  // filter op
  auto filterOp = op.getFilter().getDefiningOp<top::WeightOp>();
  auto filter_type = module::getStorageType(op.getFilter());
  std::vector<int64_t> filter_shape = {1, attr.oc, attr.ic / attr.g,
                                       attr.kh * attr.kw};
  auto new_filter_type = RankedTensorType::get(filter_shape, filter_type);
  op.getFilter().setType(new_filter_type);

  // bias op
  if (attr.with_bias) {
    auto biasOp = op.getBias().getDefiningOp<top::WeightOp>();
    auto bias_type = module::getStorageType(op.getBias());
    int64_t bias_shape[4] = {1, attr.oc, 1, 1};
    auto new_bias_type = RankedTensorType::get(bias_shape, bias_type);
    op.getBias().setType(new_bias_type);
  }
  return success();
}

// ======================================
// GlobalGenInterface
// ======================================

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  /* common param */
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long output_global_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];     // (kh, kw)
  int stride[2];     // (h, w)
  int dilation[2];   // (h, w)
  int pad[4];        // (h0, h1, w0, w1)
  int output_pad[2]; // (h, w)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  unsigned long long kzp_global_addr;
  unsigned long long pad_insert_global_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv_global_param_t;

typedef struct {
  /* common param */
  unsigned int input_local_addr;
  unsigned int weight_local_addr;
  unsigned int bias_local_addr;
  unsigned int buffer_local_addr;
  unsigned int output_local_addr;
  int input_shape[4];
  int groups;
  int output_c;
  int kernel[2];   // (kh, kw)
  int stride[2];   // (h, w)
  int dilation[2]; // (h, w)
  int pad[4];      // (h0, h1, w0, w1)
  int has_bias;
  int input_dtype;
  int weight_dtype;
  int bias_dtype;
  /* param for float */
  int output_dtype;
  int if_relu;
  float upper_limit;
  /* param for quant */
  bool is_asym;
  unsigned char rshift;
  unsigned int kzp_local_addr;
  unsigned int pad_insert_local_addr;
  bool kzp_is_const;
  bool pad_insert_is_const;
  int kzp_val;
  int pad_val;
  int insert_val;
  int kzp_dtype;
} deconv_local_param_t;

#ifdef __cplusplus
}
#endif

void tpu::DeconvOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  deconv_global_param_t param = {0};
  param.input_global_addr = module::getAddress(getInput());
  param.weight_global_addr = module::getAddress(getFilter());
  param.bias_global_addr = module::getAddress(getBias());
  param.output_global_addr = module::getAddress(getOutput());
  param.input_shape[0] = attr.n;
  param.input_shape[1] = attr.ic;
  param.input_shape[2] = attr.ih;
  param.input_shape[3] = attr.iw;
  param.groups = attr.g;
  param.output_c = attr.oc;
  param.kernel[0] = attr.kh;
  param.kernel[1] = attr.kw;
  param.stride[0] = attr.sh;
  param.stride[1] = attr.sw;
  param.dilation[0] = attr.dh;
  param.dilation[1] = attr.dw;
  param.pad[0] = attr.pad_h;
  param.pad[1] = attr.pad_h_after;
  param.pad[2] = attr.pad_w;
  param.pad[3] = attr.pad_w_after;
  param.output_pad[0] = attr.output_pad_h;
  param.output_pad[1] = attr.output_pad_w;
  param.has_bias = attr.with_bias;
  param.input_dtype = BM168x::getDataType(getInput());
  param.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM168x::getDataType(getBias());
  }
  param.output_dtype = BM168x::getDataType(getOutput());
  param.if_relu = attr.do_relu;
  param.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.is_asym = true;
    param.rshift = 0;
    param.kzp_global_addr = 0;
    param.pad_insert_global_addr = 0;
    param.kzp_is_const = true;
    param.pad_insert_is_const = true;
    param.kzp_val = 0;
    param.pad_val = in_qtype.getZeroPoint();
    param.insert_val = in_qtype.getZeroPoint();
    param.kzp_dtype = param.input_dtype;
  }
  BM168x::call_global_func("backend_api_deconv_global", &param, sizeof(param));
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::DeconvOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t sz = out_lmem_bytes * sizeof(int32_t);
  auto &attr = getDeconvParam(*this);

  auto idtype = BM168x::getDataType(getInput());
  int type_len = BM168x::getFmtBytes(idtype);
  int64_t eu_num = BM168x::eu_num(type_len);

  int ic_per_npu = ceiling_func(attr.ic / attr.g, BM168x::NPU_NUM);
  // fp part 2: used for group > 1, input must start from npu 0
  if (attr.g > 1 &&
      (idtype == DTYPE_FP32 || idtype == DTYPE_BFP16 || idtype == DTYPE_FP16)) {
    sz += ic_per_npu * align_up(in_hslice * attr.iw, eu_num) * type_len;
  }
  // quant : used for groups > 1, input must start from npu 0,
  if (attr.g > 1 && !attr.is_dw && type_len == 1) {
    sz +=
        ic_per_npu *
        (align_up(in_hslice * attr.iw, eu_num) + attr.pad_insert_is_const ? 0
                                                                          : 2);
  }

  return sz;
}

void tpu::DeconvOp::assign_sec_info(int64_t n_step, int64_t h_step,
                                    void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  memset(sec_info, 0, sizeof(local_sec_info_t));

  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  sec_info->n_slice = in_gi.n_slice;
  sec_info->d_slice = 1;
  sec_info->h_slice = in_gi.h_slice;
  sec_info->h_idx = in_gi.h_idx;
  sec_info->is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attr.ih);
  sec_info->w_slice = attr.iw;
  sec_info->out_n_slice = gi.n_slice;
  sec_info->out_h_idx = gi.h_idx;
  sec_info->out_h_slice = gi.h_slice;
  sec_info->out_w_slice = attr.ow;
}

void tpu::DeconvOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                          void *sec_info_) {
  local_sec_info_t *sec_info = (local_sec_info_t *)sec_info_;
  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  auto filter_gi = LocalGenInterface::getGroupInfo(getFilter(), n_step, h_step);
  auto bias_gi = LocalGenInterface::getGroupInfo(getBias(), n_step, h_step);

  deconv_local_param_t param = {0};
  param.input_local_addr = (uint32_t)in_gi.out_addr;
  param.weight_local_addr = (uint32_t)filter_gi.out_addr;
  param.bias_local_addr = (uint32_t)bias_gi.out_addr;
  param.output_local_addr = (uint32_t)gi.out_addr;
  param.buffer_local_addr = gi.buffer_addr;
  param.input_shape[0] = sec_info->n_slice;
  param.input_shape[1] = attr.ic;
  param.input_shape[2] = sec_info->h_slice;
  param.input_shape[3] = attr.iw;
  param.groups = attr.g;
  param.output_c = attr.oc;
  param.kernel[0] = attr.kh;
  param.kernel[1] = attr.kw;
  param.stride[0] = attr.sh;
  param.stride[1] = attr.sw;
  param.dilation[0] = attr.dh;
  param.dilation[1] = attr.dw;
  int kh_ext = (attr.kh - 1) * attr.dh + 1;
  if (auto deconv_in_slice =
          DeconvSlice(gi.h_idx, gi.h_slice, attr.sh, kh_ext, attr.pad_h)) {
    param.pad[0] = deconv_in_slice.value()[0];
    param.pad[1] = deconv_in_slice.value()[1];
  } else {
    param.pad[0] = attr.kh - attr.pad_h - 1;
    param.pad[1] = attr.kh - attr.pad_h_after - 1;
  }
  param.pad[2] = attr.kw - attr.pad_w - 1;
  param.pad[3] = attr.kw - attr.pad_w_after - 1;
  param.has_bias = attr.with_bias;
  param.input_dtype = BM168x::getDataType(getInput());
  param.weight_dtype = BM168x::getDataType(getFilter());
  if (getBias().getType().isa<RankedTensorType>()) {
    param.bias_dtype = BM168x::getDataType(getBias());
  }

  param.output_dtype = BM168x::getDataType(getOutput());
  param.if_relu = attr.do_relu;
  param.upper_limit = attr.relu_limit;
  if (module::isUniformQuantized(getInput())) {
    auto in_qtype = module::getUniformQuantizedType(getInput());
    param.is_asym = true;
    param.rshift = 0;
    param.kzp_local_addr = 0;
    param.pad_insert_local_addr = 0;
    param.kzp_is_const = true;
    param.pad_insert_is_const = true;
    param.kzp_val = 0;
    param.pad_val = in_qtype.getZeroPoint();
    param.insert_val = in_qtype.getZeroPoint();
    param.kzp_dtype = param.weight_dtype;
  }
  BM168x::call_local_func("backend_api_deconv_local", &param, sizeof(param));
}
