#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/MathUtils.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

// ======================================
// WeightReorderInterface
// ======================================

// convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
template <typename T>
static void
reshape_coeff_for_broadcast_channel(std::shared_ptr<std::vector<T>> &coeff,
                                    std::vector<int64_t> &shape,
                                    bool align = false) {
  int64_t n, c, h, w;
  Module::getNCHW(shape, n, c, h, w);
  if (n != 1 || h != 1 || c <= BM1686::NPU_NUM) {
    return;
  }
  // convert (1, oc, 1, w) to (1, NPU_NUM, 1, DIV_UP(oc, NPU_NUM) * w)
  int64_t new_c = BM1686::NPU_NUM;
  int type_len = sizeof(T);
  auto c2w = ceiling_func(c, new_c);
  auto old_w_align = align_up(w, BM1686::instance().get_eu_num(type_len));
  int new_w = (align ? old_w_align : w) * (c2w - 1) + w;
  int64_t new_size = new_w * new_c * type_len;
  auto filter_new = std::make_shared<std::vector<T>>(new_size, 0);
  for (int i = 0; i < c2w; i++) {
    for (int j = 0; j < new_c; j++) {
      for (int k = 0; k < w; k++) {
        int src_idx = i * new_c * w + j * w + k;
        int dst_idx = j * new_w + i * (align ? old_w_align : w) + k;
        filter_new->at(dst_idx) = coeff->at(src_idx);
      }
    }
  }
  shape = {1, new_c, 1, new_w};
  coeff = filter_new;
}

// refer to net_compiler: bool BM1686CoeffArranger::ConvWeightArr(GraphEdge*
// edge)
void tpu::ConvOp::weight_reorder_int8_bm1686() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  if (is_dw) {
    llvm_unreachable("depthwise should support !!");
  }
  auto elem_type = Module::getStorageType(filter());
  int64_t IC_PARALLEL = 64;
  int64_t merge_w = 0;
  auto type_bytes = elem_type.getIntOrFloatBitWidth() / 8;
  size_t new_bytes = align_up(ic, IC_PARALLEL) * oc * kh * kw * type_bytes;
  auto filter_i8 = filterOp.read_as_byte();
  auto filter_new = std::make_shared<std::vector<uint8_t>>(new_bytes, 0);
  auto kernel_hw = kh * kw;
  int64_t new_ic = ceiling_func(ic, IC_PARALLEL);
  int64_t new_hw = kernel_hw * IC_PARALLEL;
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
          filter_new->at(trans_offset) = filter_i8->at(orig_offset);
        }
      }
    }
  }
  auto filter_type = filterOp.getType().cast<RankedTensorType>();
  std::vector<int64_t> filter_shape = {1, oc, 1, new_ic * new_hw};
  // refer to net_compier: reshape_coeff_for_broadcast_channel(weight, false);
  reshape_coeff_for_broadcast_channel(filter_new, filter_shape);

  auto filter_w_bytes = filter_shape[3];
  merge_w += filter_w_bytes;
  std::shared_ptr<std::vector<int32_t>> bias_new;
  std::vector<int64_t> bias_shape = {1, oc, 1, 1};
  int64_t bias_w_bytes = 0;
  if (with_bias) {
    auto biasOp = bias().getDefiningOp<top::WeightOp>();
    bias_new = biasOp.read<int32_t>();
    reshape_coeff_for_broadcast_channel(bias_new, bias_shape, false);
    bias_w_bytes = bias_shape[3] * sizeof(int32_t);
    merge_w += bias_w_bytes;
  }

  // add requant op
  auto op = getOperation();
  auto qtype = Quant::getUniformQuantizedType(output());
  std::vector<int64_t> quant_shape = {1, oc, 1, 3};
  auto quant_data = std::make_shared<std::vector<int32_t>>(oc * 3, 0);
  auto m_data = Module::getI64Array(multiplier(), oc, 1);
  auto r_data = Module::getI64Array(rshift(), oc, 0);
  for (int i = 0; i < oc; i++) {
    quant_data->at(i * 3) = m_data->at(i);
    quant_data->at(i * 3 + 1) = r_data->at(i);
    quant_data->at(i * 3 + 2) = qtype.getZeroPoint();
  }
  reshape_coeff_for_broadcast_channel(quant_data, quant_shape, true);
  auto quant_w_bytes = quant_shape[3] * sizeof(int32_t);
  merge_w += quant_w_bytes;
  // merge requant/bias/filter
  auto new_coeff =
      std::make_shared<std::vector<int8_t>>(BM1686::NPU_NUM * merge_w, 0);
  std::vector<int64_t> coeff_shape = {1, BM1686::NPU_NUM, 1, merge_w};
  for (int i = 0; i < BM1686::NPU_NUM; i++) {
    auto coeff_ptr = new_coeff->data() + i * merge_w;
    auto quant_ptr = quant_data->data() + i * quant_shape[3];
    auto bias_ptr = with_bias ? bias_new->data() + i * bias_shape[3] : nullptr;
    auto filter_ptr = filter_new->data() + i * filter_shape[3];
    // copy quant
    memcpy(coeff_ptr, quant_ptr, quant_w_bytes);
    coeff_ptr += quant_w_bytes;
    if (with_bias) {
      memcpy(coeff_ptr, bias_ptr, bias_w_bytes);
      coeff_ptr += bias_w_bytes;
    }
    memcpy(coeff_ptr, filter_ptr, filter_w_bytes);
  }
  OpBuilder builder(getContext());
  auto coeff_type = RankedTensorType::get(coeff_shape, builder.getI8Type());
  auto coeff_op = top::WeightOp::create(op, "merge", *new_coeff, coeff_type);
  op->removeAttr("rshift");
  op->removeAttr("multiplier");
  op->setAttr("coeff_merged", builder.getBoolAttr(true));
  op->setOperand(1, coeff_op);
  auto none = Module::getNoneOp(op);
  op->setOperand(2, none.getResult());
}

void tpu::ConvOp::weight_reorder_float_bm1686() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, relu);
  if (is_dw) {
    llvm_unreachable("depthwise should support !!");
  }
  auto out_type = Module::getStorageType(output());
  auto filterOp = filter().getDefiningOp<top::WeightOp>();
  int64_t filter_shape[4];
  if (out_type.isF32()) {
     filter_shape[0] = 1;
     filter_shape[1] = oc;
     filter_shape[2] = ic/g;
     filter_shape[3] = kh*kw;
  } else if (out_type.isBF16() || out_type.isF16()) {
    assert(g == 1); // ??
    int64_t IC_PARALLEL = 32;
    int64_t fmt_bytes = 2;
    int64_t new_bytes = align_up(ic, IC_PARALLEL) * oc * kh * kw * fmt_bytes;
    std::vector<int8_t> weight_trans(new_bytes, 0);
  } else {
    dump();
    llvm_unreachable("op type not support");
  }
}

// ======================================
// GlobalGenInterface
// ======================================

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t input_global_addr;
  uint64_t weight_global_addr;
  uint64_t bias_global_addr;
  uint64_t kzp_global_addr;
  uint64_t pad_global_addr;
  uint64_t output_global_addr;
  int batch_num;
  int input_c;
  int input_h;
  int input_w;
  int groups;
  int output_c;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
  int dh;
  int dw;
  int pad_h;
  int pad_h_after;
  int pad_w;
  int pad_w_after;
  int has_bias;
  int if_relu;
  float upper_limit;
  int rshift;
  int idtype;
  int wdtype;
  int bdtype;
  int kdtype;
  int odtype;
  int round_mode;
  /**
   * merge_coeff:
   *    0: Not merge and not reshape weight and bias
   *    1. reshape and merge weight and bias as (bias, weight) align to (4, 1)
   * bytes for depthwise_fix8b or (4, 64) bytes for conv_fix8b
   *    2. reshape and merge weight, bias and requant as has bias-(requant,
   * bias, weight) align to (64, 4, 1) bytes for depthwise_fix8b or (64, 4, 64)
   * bytes for conv_fix8b or no bias-(requant, weight) align to (64, 1) bytes
   * for depthwise_fix8b or (64, 64) bytes for conv_fix8b
   */
  int merge_coeff;
  bool is_asym;
  bool kzp_is_const;
  bool pad_is_const;
  int kzp_val;
  int pad_val;
} conv_global_param_t;

typedef struct conv_common_spec {
  int32_t groups;
  int32_t input_c;
  int32_t output_c;
  int32_t kh;
  int32_t kw;
  int32_t stride_h;
  int32_t stride_w;
  int32_t dh;
  int32_t dw;
  int32_t pad_h_t;
  int32_t pad_h_b;
  int32_t pad_w_l;
  int32_t pad_w_r;
  int32_t has_bias;
  int32_t if_relu;
  float upper_limit;
  int32_t rshift;
  int32_t round_mode;
  int32_t is_asym;
  int32_t kzp_is_const;
  int32_t kzp_value;
  int32_t ipad_is_const;
  int32_t ipad_value;
  int32_t bias_sign; // For merged coeff
} conv_common_spec_t;

typedef struct conv_global_spec {
  conv_common_spec_t common;
  /**
   * merge_coeff:
   *    0: Not merge and not reshape weight and bias
   *    1. reshape and merge weight and bias as (bias, weight) align to (4, 1)
   * bytes for depthwise_fix8b or (4, 64) bytes for conv_fix8b
   *    2. reshape and merge weight, bias and requant as has bias-(requant,
   * bias, weight) align to (64, 4, 1) bytes for depthwise_fix8b or (64, 4, 64)
   * bytes for conv_fix8b or no bias-(requant, weight) align to (64, 1) bytes
   * for depthwise_fix8b or (64, 64) bytes for conv_fix8b
   */
  int32_t merge_coeff;
  int32_t weight_is_tensor;
} conv_global_spec_t;

typedef struct conv_local_spec {
  conv_common_spec_t common;
  uint32_t buffer_local_addr;
  int32_t result_add;
  int32_t unused_ht_for_input;
  int32_t unused_hb_for_input;
  int32_t unused_wl_for_input;
  int32_t unused_wr_for_input;
  int32_t use_3ic_optimize;
  int32_t group_one_conv;
  int32_t with_requant;
  int32_t merge_coeff;

  // For dynamic inference
  uint32_t concat_c;
  int32_t concat_c_idx;
  int32_t reference_id;
} conv_local_spec_t;

typedef struct conv_local_param {
  conv_local_spec_t spec;
} conv_local_param_t;

#ifdef __cplusplus
}
#endif

void tpu::ConvOp::codegen_global_int8_bm1686() {
  //   conv_global_param_t param = {0};
  //   int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt,
  //   pb,
  //       pl, pr, dh, dw;
  //   bool is_dw, with_bias, do_relu;
  //   parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw,
  //   pt, pb,
  //              pl, pr, dh, dw, is_dw, with_bias, do_relu);
  //   param.input_global_addr = Module::getAddress(input());
  //   param.weight_global_addr = Module::getAddress(filter());
  //   param.output_global_addr = Module::getAddress(output());
  //   param.has_bias = with_bias;
  //   param.batch_num = n;
  //   param.input_c = ic;
  //   param.input_h = ih;
  //   param.input_w = iw;
  //   param.groups = g;
  //   param.output_c = oc;
  //   param.if_relu = do_relu;
  //   param.upper_limit = 0;
  //   param.idtype = BM168x::getDataType(input());
  //   param.wdtype = BM168x::getDataType(filter());
  //   param.merge_coeff = 2;
  //   param.bdtype = DTYPE_INT32;
  //   param.bias_global_addr = 0;
  //   param.odtype = BM168x::getDataType(output());
  //   param.kh = kh;
  //   param.kw = kw;
  //   param.dh = dh;
  //   param.dw = dw;
  //   param.stride_h = sh;
  //   param.stride_w = sw;
  //   param.pad_h = pt;
  //   param.pad_h_after = pb;
  //   param.pad_w = pl;
  //   param.pad_w_after = pr;
  //   param.rshift = 0;
  //   param.round_mode = ROUND_UP;
  //   param.is_asym = true;
  //   param.kdtype = DTYPE_INT8;
  //   param.kzp_global_addr = 0;
  //   param.pad_global_addr = 0;
  //   param.kzp_is_const = 1;
  //   param.kzp_val = 0;
  //   param.pad_is_const = 1;
  //   auto input_type = Quant::getUniformQuantizedType(input());
  //   param.pad_val = input_type.getZeroPoint();
  //   BM1686::instance().call_global_func("backend_api_conv_global", &param,
  //                                       sizeof(conv_global_param_t));
}

// f32
void tpu::ConvOp::codegen_global_float_bm1686() {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, do_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, do_relu);
  auto op = getOperation();
  auto input_spec = BM1686::get_input_global_spec(op);
  auto output_spec = BM1686::get_output_global_spec(op);
  conv_global_spec_t spec = {0};
  auto &common = spec.common;
  common.input_c = ic;
  common.output_c = oc;
  common.if_relu = do_relu;
  common.upper_limit = 0;
  common.kh = kh;
  common.kw = kw;
  common.dh = dh;
  common.dw = dw;
  common.stride_h = sh;
  common.stride_w = sw;
  common.groups = g;
  common.pad_h_t = pt;
  common.pad_h_b = pb;
  common.pad_w_l = pl;
  common.pad_w_r = pr;
  common.round_mode = ROUND_UP;
  common.has_bias = with_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.ipad_value = 0;
  BM1686::instance().call_global_func("backend_api_conv_global", &spec,
                                      sizeof(spec), input_spec->data(),
                                      output_spec->data());
}

// ======================================
// LocalGenInterface
// ======================================

int64_t tpu::ConvOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                          int64_t out_h, int64_t out_w,
                                          int64_t out_lmem_bytes) {
  if (coeff_merged() == false) {
    return 0;
  }
  return out_lmem_bytes * sizeof(int32_t);
}

void tpu::ConvOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  // int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
  //     pl, pr, dh, dw;
  // bool is_dw, with_bias, do_relu;
  // parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt,
  // pb,
  //            pl, pr, dh, dw, is_dw, with_bias, do_relu);
  // auto in_ginfo = LocalGenInterface::getGroupInfo(input());
  // auto weight_ginfo = LocalGenInterface::getGroupInfo(filter());
  // auto out_ginfo = LocalGenInterface::getGroupInfo(output());
  // conv_local_param_t param;
  // memset(&param, 0, sizeof(param));
  // auto &p = param.spec.common;
  // p.has_bias = with_bias;
  // p.input_local_addr = in_ginfo.out_addr;
  // p.weight_local_addr = weight_ginfo.out_addr;
  // param.output_local_addr = out_ginfo.out_addr;
  // param.buffer_local_addr = out_ginfo.buffer_addr;
  // param.batch_num = in_ginfo.n_slice;
  // param.input_c = ic;
  // param.input_h = in_ginfo.h_slice;
  // param.input_w = iw;
  // param.output_c = oc;
  // param.if_relu = do_relu;
  // param.upper_limit = 0;
  // param.result_add = 0;
  // param.idtype = BM168x::getDataType(input());
  // param.wdtype = BM168x::getDataType(filter());
  // param.with_requant = true;
  // param.odtype = BM168x::getDataType(output());
  // param.kh = kh;
  // param.kw = kw;
  // param.dh = dh;
  // param.dw = dw;
  // param.stride_h = sh;
  // param.stride_w = sw;
  // param.groups = g;
  // param.rshift = 0;
  // param.pad_h_t = (in_ginfo.h_idx == 0 ? pt : 0);
  // param.pad_h_b = (in_ginfo.h_idx + in_ginfo.h_slice == ih ? pb : 0);
  // param.pad_w_l = pl;
  // param.pad_w_r = pr;
  // param.unused_ht_for_input = 0, param.unused_hb_for_input = 0,
  // param.unused_wl_for_input = 0, param.unused_wr_for_input = 0,
  // param.use_3ic_optimize = 0;
  // param.group_one_conv = false;
  // param.round_mode = ROUND_UP;
  // param.is_asym = true;

  // bool is_depthwise = (param.groups > 1 && param.groups == param.input_c &&
  //                      param.groups == param.output_c);
  // param.bdtype = DTYPE_INT32;
  // param.bias_local_addr = param.weight_local_addr;
  // int rq_wsize = (ceiling_func(param.output_c, (int)BM1686::NPU_NUM) - 1) *
  //                    BM1686::EU_BYTES +
  //                3 * sizeof(int32_t);
  // if (param.has_bias) {
  //   int bias_wsize =
  //       ceiling_func(param.output_c, (int)BM1686::NPU_NUM) * sizeof(int32_t);
  //   int offset = is_depthwise ? rq_wsize + bias_wsize
  //                             : align_up(rq_wsize + bias_wsize, 64);
  //   param.weight_local_addr = param.weight_local_addr + offset;
  // } else {
  //   int offset = is_depthwise ? rq_wsize : align_up(rq_wsize, 64);
  //   param.weight_local_addr = param.weight_local_addr + offset;
  // }
  // BM1686::instance().call_local_func("backend_api_conv_local", &param,
  //                                    sizeof(conv_local_param_t));
}

void tpu::ConvOp::codegen_local_float_bm1686(int64_t n_step, int64_t h_step) {
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, do_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, do_relu);
  auto op = getOperation();
  auto input_spec = BM1686::get_input_local_spec(op);
  auto output_spec = BM1686::get_output_local_spec(op);
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  conv_local_param_t p;
  memset(&p, sizeof(p), 0);
  p.spec.buffer_local_addr = gi.buffer_addr;
  auto &common = p.spec.common;
  common.input_c = ic;
  common.output_c = oc;
  common.if_relu = do_relu;
  common.upper_limit = 0;
  common.kh = kh;
  common.kw = kw;
  common.dh = dh;
  common.dw = dw;
  common.stride_h = sh;
  common.stride_w = sw;
  common.groups = g;
  common.pad_h_t = pt; // judge inner ? (in_gi.h_idx == 0 ? pt : 0);
  common.pad_h_b = pb; // (in_gi.h_idx + in_gi.h_slice == ih ? pb : 0);
  common.pad_w_l = pl;
  common.pad_w_r = pr;
  common.round_mode = ROUND_UP;
  common.has_bias = with_bias;
  common.bias_sign = true;
  common.ipad_is_const = true;
  common.ipad_value = 0;
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split =
      (in_gi.h_idx == 0 && (in_gi.h_idx + in_gi.h_slice) == ih);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_slice = gi.h_slice;
  BM1686::instance().call_local_func("backend_api_conv_local", &p, sizeof(p),
                                     &sec_info, input_spec->data(),
                                     output_spec->data());
}
