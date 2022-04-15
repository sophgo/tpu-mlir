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
  conv_global_param_t param = {0};
  int64_t n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
      pl, pr, dh, dw;
  bool is_dw, with_bias, do_relu;
  parseParam(n, ic, ih, iw, oc, oh, ow, g, kh, kw, ins_h, ins_w, sh, sw, pt, pb,
             pl, pr, dh, dw, is_dw, with_bias, do_relu);
  param.input_global_addr = Module::getAddress(input());
  param.weight_global_addr = Module::getAddress(filter());
  param.output_global_addr = Module::getAddress(output());
  param.has_bias = with_bias;
  param.batch_num = n;
  param.input_c = ic;
  param.input_h = ih;
  param.input_w = iw;
  param.groups = g;
  param.output_c = oc;
  param.if_relu = do_relu;
  param.upper_limit = 0;
  param.idtype = BM168x::getDataType(input());
  param.wdtype = BM168x::getDataType(filter());
  param.merge_coeff = 2;
  param.bdtype = DTYPE_INT32;
  param.bias_global_addr = 0;
  param.odtype = BM168x::getDataType(output());
  param.kh = kh;
  param.kw = kw;
  param.dh = dh;
  param.dw = dw;
  param.stride_h = sh;
  param.stride_w = sw;
  param.pad_h = pt;
  param.pad_h_after = pb;
  param.pad_w = pl;
  param.pad_w_after = pr;
  param.rshift = 0;
  param.round_mode = ROUND_UP;
  param.is_asym = true;
  param.kdtype = DTYPE_INT8;
  param.kzp_global_addr = 0;
  param.pad_global_addr = 0;
  param.kzp_is_const = 1;
  param.kzp_val = 0;
  param.pad_is_const = 1;
  auto input_type =
      Quant::getQuantizedType<quant::UniformQuantizedType>(input());
  param.pad_val = input_type.getZeroPoint();
  BM1686::instance().call_global_func("backend_api_conv_global", &param,
                                      sizeof(conv_global_param_t));
}
