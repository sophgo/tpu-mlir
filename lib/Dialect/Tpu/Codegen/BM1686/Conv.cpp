#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/MathUtils.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

typedef struct {
  unsigned long long input_global_addr;
  unsigned long long weight_global_addr;
  unsigned long long bias_global_addr;
  unsigned long long kzp_global_addr;
  unsigned long long pad_global_addr;
  unsigned long long output_global_addr;
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
     *    1. reshape and merge weight and bias as (bias, weight) align to (4, 1) bytes for depthwise_fix8b or (4, 64) bytes for conv_fix8b
     *    2. reshape and merge weight, bias and requant as has bias-(requant, bias, weight) align to (64, 4, 1) bytes for depthwise_fix8b or (64, 4, 64) bytes for conv_fix8b
     *                                                   or no bias-(requant, weight) align to (64, 1) bytes for depthwise_fix8b or (64, 64) bytes for conv_fix8b
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

void tpu::ConvOp::codegen_global_int8_bm1686() {
//   conv_global_param_t param = {0};
//   int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
//       pl, pr, dh, dw;
//   bool is_dw, with_bias, do_relu;
//   parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
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

void tpu::ConvOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  // int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
  //     pl, pr, dh, dw;
  // bool is_dw, with_bias, do_relu;
  // parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
  //            pl, pr, dh, dw, is_dw, with_bias, do_relu);
  // auto in_ginfo = LayerGroupInterface::getGroupInfo(input());
  // auto weight_ginfo = LayerGroupInterface::getGroupInfo(filter());
  // auto out_ginfo = LayerGroupInterface::getGroupInfo(output());
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
