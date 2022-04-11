#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

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

void tpu::ConvOp::codegen_int8_bm1686() {
  // conv_global_param_t param = {0};
  // int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
  //     pl, pr, dh, dw;
  // bool is_dw, with_bias, do_relu;
  // parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
  //            pl, pr, dh, dw, is_dw, with_bias, do_relu);
  // param.input_global_addr = Module::getAddress(input());
  // param.weight_global_addr = Module::getAddress(filter());
  // param.output_global_addr = Module::getAddress(output());
  // param.has_bias = with_bias;
  // param.batch_num = n;
  // param.input_c = ic;
  // param.input_h = ih;
  // param.input_w = iw;
  // param.groups = g;
  // param.output_c = oc;
  // param.if_relu = do_relu;
  // param.upper_limit = 0;
  // param.idtype = BM168x::getDataType(input());
  // param.wdtype = BM168x::getDataType(filter());
  // param.merge_coeff = layer_param->layer_param_u.conv_param.merge_weight_bias;
  // param.if_relu = lp.with_requant
  //                     ? !net_graph_->get_tensor_sign(layer_out_tensors[0])
  //                     : param.if_relu;
  // if (param.has_bias && param.merge_coeff == 0) {
  //   param.bdtype = net_graph_->get_tensor_data_type(layer_in_tensors[2]);
  //   param.bias_global_addr =
  //       net_graph_->get_tensor_global_mem(layer_in_tensors[2])->addr;
  // } else if (param.merge_coeff == 2) {
  //   param.bdtype = lp.bias_sign ? DTYPE_INT32 : DTYPE_UINT32;
  //   param.bias_global_addr = 0;
  // }
  // param.odtype = net_graph_->get_tensor_data_type(layer_out_tensors[0]);
  // param.kh = kh;
  // param.kw = kw;
  // param.dh = dh;
  // param.dw = dw;
  // param.stride_h = sh;
  // param.stride_w = sw;
  // param.pad_h = lp.pad_h;
  // param.pad_h_after = lp.pad_h_after;
  // param.pad_w = lp.pad_w;
  // param.pad_w_after = lp.pad_w_after;
  // param.rshift = lp.rshift_num;
  // param.round_mode = ROUND_UP;
  // param.is_asym = lp.is_asym;
  // if (lp.is_asym) {
  //   param.kdtype = DTYPE_UINT8;
  //   if (lp.kzp_is_const && lp.kzp_value < 0) {
  //     param.kdtype = DTYPE_INT8;
  //   } else if (!lp.kzp_is_const) {
  //     net_graph_->get_tensor_data_type(layer_in_tensors[param.has_bias + 2]);
  //   }
  //   param.kzp_global_addr =
  //       lp.kzp_is_const
  //           ? 0
  //           : net_graph_
  //                 ->get_tensor_global_mem(layer_in_tensors[param.has_bias + 2])
  //                 ->addr;
  //   param.pad_global_addr =
  //       lp.ipad_is_const
  //           ? 0
  //           : net_graph_
  //                 ->get_tensor_global_mem(
  //                     layer_in_tensors[param.has_bias + !lp.kzp_is_const + 2])
  //                 ->addr;
  //   param.kdtype = lp.kzp_is_const ? DTYPE_INT8
  //                                  : net_graph_->get_tensor_data_type(
  //                                        layer_in_tensors[param.has_bias + 2]);
  //   param.kzp_is_const = lp.kzp_is_const;
  //   param.kzp_val = lp.kzp_value;
  //   param.pad_is_const = lp.ipad_is_const;
  //   param.pad_val = lp.ipad_value;
  // }
  // call_global_func("backend_api_conv_global", &param,
  //                  sizeof(conv_global_param_t));
}
