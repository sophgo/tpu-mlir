#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

typedef struct {
  bool is_local;
  /* for common */
  unsigned long long input_addr;
  unsigned long long output_addr;
  const int *input_shape;
  int output_h;
  int output_w;
  int kh;
  int kw;
  int pad_h_t;
  int pad_h_b;
  int pad_w_l;
  int pad_w_r;
  int stride_h;
  int stride_w;
  int dh;
  int dw;
  int is_avg_pooling;
  int avg_pooling_mode;
  DATA_TYPE_T idtype;
  DATA_TYPE_T odtype;
  /* for float */
  int if_relu;
  float relu_upper_limit;
  /* for fix8b */
  int ceil_mode;
  ROUND_MODE_T round_mode;
} pooling_param_t;

void tpu::MaxPoolOp::codegen_int8_bm1686() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  int input_shape[4] = {(int)n, (int)c, (int)ih, (int)iw};
  pooling_param_t p = {0};
  p.is_local = false;
  p.input_addr = Module::getAddress(input());
  p.output_addr = Module::getAddress(output());
  p.input_shape = input_shape;
  p.output_h = oh;
  p.output_w = ow;
  p.kh = kh;
  p.kw = kw;
  p.pad_h_t = pt;
  p.pad_h_b = pb;
  p.pad_w_l = pl;
  p.pad_w_r = pr;
  p.stride_h = sh;
  p.stride_w = sw;
  p.dh = 1;
  p.dw = 1;
  p.is_avg_pooling = 0;
  p.avg_pooling_mode = 0;
  p.idtype = BM168x::getDataType(input());
  p.odtype = BM168x::getDataType(output());
  p.if_relu = relu;
  p.relu_upper_limit = 0;
  p.ceil_mode = 0;
  p.round_mode = ROUND_UP;
  BM1686::instance().call_global_func("backend_api_pooling", &p,
                                      sizeof(pooling_param_t));
}

void tpu::AvgPoolOp::codegen_int8_bm1686() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  int input_shape[4] = {(int)n, (int)c, (int)ih, (int)iw};
  pooling_param_t p = {0};
  p.is_local = false;
  p.input_addr = Module::getAddress(input());
  p.output_addr = Module::getAddress(output());
  p.input_shape = input_shape;
  p.output_h = oh;
  p.output_w = ow;
  p.kh = kh;
  p.kw = kw;
  p.pad_h_t = pt;
  p.pad_h_b = pb;
  p.pad_w_l = pl;
  p.pad_w_r = pr;
  p.stride_h = sh;
  p.stride_w = sw;
  p.dh = 1;
  p.dw = 1;
  p.is_avg_pooling = 1;
  p.avg_pooling_mode = count_include_pad ? 0 : 1;
  p.idtype = BM168x::getDataType(input());
  p.odtype = BM168x::getDataType(output());
  p.if_relu = relu;
  p.relu_upper_limit = 0;
  p.ceil_mode = 0;
  p.round_mode = ROUND_UP;
  BM1686::instance().call_global_func("backend_api_pooling", &p,
                                      sizeof(pooling_param_t));
}
