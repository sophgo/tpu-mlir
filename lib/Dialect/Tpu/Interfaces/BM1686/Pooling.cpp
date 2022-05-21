#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct pooling_common_spec {
  int32_t kh;
  int32_t kw;
  int32_t pad_h_t;
  int32_t pad_h_b;
  int32_t pad_w_l;
  int32_t pad_w_r;
  int32_t stride_h;
  int32_t stride_w;
  int32_t dh;
  int32_t dw;
  int32_t is_global_pooling;
  int32_t is_avg_pooling;
  int32_t avg_pooling_mode;
  /* for float */
  int32_t if_relu;
  float relu_upper_limit;
  /* for fix8b */
  int32_t ceil_mode;
  int32_t round_mode;
} pooling_common_spec_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================
// int8
void tpu::MaxPoolOp::codegen_global_int8_bm1686() {
  // int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  // bool relu, is_global, count_include_pad;
  // parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
  //            relu, is_global, count_include_pad);
  // int input_shape[4] = {(int)n, (int)c, (int)ih, (int)iw};
  // pooling_param_t p = {0};
  // p.is_local = false;
  // p.input_addr = Module::getAddress(input());
  // p.output_addr = Module::getAddress(output());
  // p.input_shape = input_shape;
  // p.output_h = oh;
  // p.output_w = ow;
  // p.kh = kh;
  // p.kw = kw;
  // p.pad_h_t = pt;
  // p.pad_h_b = pb;
  // p.pad_w_l = pl;
  // p.pad_w_r = pr;
  // p.stride_h = sh;
  // p.stride_w = sw;
  // p.dh = 1;
  // p.dw = 1;
  // p.is_avg_pooling = 0;
  // p.avg_pooling_mode = 0;
  // p.idtype = BM168x::getDataType(input());
  // p.odtype = BM168x::getDataType(output());
  // p.if_relu = relu;
  // p.relu_upper_limit = 0;
  // p.ceil_mode = 0;
  // p.round_mode = ROUND_UP;
  // BM1686::instance().call_global_func("backend_api_pooling", &p,
  //                                     sizeof(pooling_param_t));
}

void tpu::AvgPoolOp::codegen_global_int8_bm1686() {
  // int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  // bool relu, is_global, count_include_pad;
  // parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
  //            relu, is_global, count_include_pad);
  // int input_shape[4] = {(int)n, (int)c, (int)ih, (int)iw};
  // pooling_param_t p = {0};
  // p.is_local = false;
  // p.input_addr = Module::getAddress(input());
  // p.output_addr = Module::getAddress(output());
  // p.input_shape = input_shape;
  // p.output_h = oh;
  // p.output_w = ow;
  // p.kh = kh;
  // p.kw = kw;
  // p.pad_h_t = pt;
  // p.pad_h_b = pb;
  // p.pad_w_l = pl;
  // p.pad_w_r = pr;
  // p.stride_h = sh;
  // p.stride_w = sw;
  // p.dh = 1;
  // p.dw = 1;
  // p.is_avg_pooling = 1;
  // p.avg_pooling_mode = count_include_pad ? 0 : 1;
  // p.idtype = BM168x::getDataType(input());
  // p.odtype = BM168x::getDataType(output());
  // p.if_relu = relu;
  // p.relu_upper_limit = 0;
  // p.ceil_mode = 0;
  // p.round_mode = ROUND_UP;
  // BM1686::instance().call_global_func("backend_api_pooling", &p,
  //                                     sizeof(pooling_param_t));
}

// f32
void tpu::AvgPoolOp::codegen_global_float_bm1686() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  auto op = getOperation();
  auto input_spec = BM1686::get_input_global_spec(op);
  auto output_spec = BM1686::get_output_global_spec(op);
  pooling_common_spec_t spec;
  spec.kh = kh;
  spec.kw = kw;
  spec.pad_h_t = pt;
  spec.pad_h_b = pb;
  spec.pad_w_l = pl;
  spec.pad_w_r = pr;
  spec.stride_h = sh;
  spec.stride_w = sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = is_global;
  spec.is_avg_pooling = true;
  spec.avg_pooling_mode = count_include_pad ? 0 : 1;
  spec.if_relu = relu;
  spec.relu_upper_limit = 0;
  spec.ceil_mode = 0;
  spec.round_mode = ROUND_UP;
  BM1686::instance().call_global_func("backend_api_pooling_global", &spec,
                                      sizeof(spec), input_spec->data(),
                                      output_spec->data());
}

void tpu::MaxPoolOp::codegen_global_float_bm1686() {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  auto op = getOperation();
  auto input_spec = BM1686::get_input_global_spec(op);
  auto output_spec = BM1686::get_output_global_spec(op);
  pooling_common_spec_t spec;
  spec.kh = kh;
  spec.kw = kw;
  spec.pad_h_t = pt;
  spec.pad_h_b = pb;
  spec.pad_w_l = pl;
  spec.pad_w_r = pr;
  spec.stride_h = sh;
  spec.stride_w = sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = is_global;
  spec.is_avg_pooling = false;
  spec.avg_pooling_mode = count_include_pad ? 0 : 1;
  spec.if_relu = relu;
  spec.relu_upper_limit = 0;
  spec.ceil_mode = 0;
  spec.round_mode = ROUND_UP;
  BM1686::instance().call_global_func("backend_api_pooling_global", &spec,
                                      sizeof(spec), input_spec->data(),
                                      output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AvgPoolOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                             int64_t out_h, int64_t out_w,
                                             int64_t out_lmem_bytes) {
  return 0;
}

int64_t tpu::MaxPoolOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                             int64_t out_h, int64_t out_w,
                                             int64_t out_lmem_bytes) {
  return 0;
}

void tpu::MaxPoolOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  // llvm_unreachable("support later");
}

void tpu::AvgPoolOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  // llvm_unreachable("support later");
}

void tpu::MaxPoolOp::codegen_local_float_bm1686(int64_t n_step,
                                                int64_t h_step) {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  auto op = getOperation();
  auto input_spec = BM1686::get_input_local_spec(op);
  auto output_spec = BM1686::get_output_local_spec(op);
  pooling_common_spec_t spec;
  spec.kh = kh;
  spec.kw = kw;
  spec.pad_h_t = pt;
  spec.pad_h_b = pb;
  spec.pad_w_l = pl;
  spec.pad_w_r = pr;
  spec.stride_h = sh;
  spec.stride_w = sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = is_global;
  spec.is_avg_pooling = false;
  spec.avg_pooling_mode = count_include_pad ? 0 : 1;
  spec.if_relu = relu;
  spec.relu_upper_limit = 0;
  spec.ceil_mode = 0;
  spec.round_mode = ROUND_UP;
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split =
      (in_gi.h_idx == 0 && (in_gi.h_idx + in_gi.h_slice) == ih);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_slice = gi.h_slice;
  BM1686::instance().call_local_func("backend_api_pooling_local", &spec,
                                     sizeof(spec), &sec_info,
                                     input_spec->data(), output_spec->data());
}

void tpu::AvgPoolOp::codegen_local_float_bm1686(int64_t n_step,
                                                int64_t h_step) {
  int64_t n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value;
  bool relu, is_global, count_include_pad;
  parseParam(n, c, ih, iw, oh, ow, kh, kw, sh, sw, pt, pb, pl, pr, pad_value,
             relu, is_global, count_include_pad);
  auto op = getOperation();
  auto input_spec = BM1686::get_input_local_spec(op);
  auto output_spec = BM1686::get_output_local_spec(op);
  pooling_common_spec_t spec;
  spec.kh = kh;
  spec.kw = kw;
  spec.pad_h_t = pt;
  spec.pad_h_b = pb;
  spec.pad_w_l = pl;
  spec.pad_w_r = pr;
  spec.stride_h = sh;
  spec.stride_w = sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = is_global;
  spec.is_avg_pooling = true;
  spec.avg_pooling_mode = count_include_pad ? 0 : 1;
  spec.if_relu = relu;
  spec.relu_upper_limit = 0;
  spec.ceil_mode = 0;
  spec.round_mode = ROUND_UP;
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(sec_info));
  sec_info.n_slice = in_gi.n_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split =
      (in_gi.h_idx == 0 && (in_gi.h_idx + in_gi.h_slice) == ih);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_slice = gi.h_slice;
  BM1686::instance().call_local_func("backend_api_pooling_local", &spec,
                                     sizeof(spec), &sec_info,
                                     input_spec->data(), output_spec->data());
}
