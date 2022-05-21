#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
} requant_fp_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t dequant_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
} dequant_fp_param_t;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::CastOp::codegen_global_int8_bm1686() {
  bool qInput = Quant::isUniformQuantized(input());
  bool qOutput = Quant::isUniformQuantized(output());
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  if (!qInput && qOutput) {
    auto qtype = Quant::getUniformQuantizedType(output());
    requant_fp_param_t param = {0};
    param.input_addr = Module::getAddress(input());
    param.output_addr = Module::getAddress(output());
    param.n = (int)n;
    param.c = (int)c;
    param.h = (int)h;
    param.w = (int)w;
    param.is_perchannel = false;
    param.scale_value = 1.0 / qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    param.output_dtype = BM168x::getDataType(output());
    param.mode = 0;
    BM1686::instance().call_global_func("backend_api_requant_float_global",
                                        &param, sizeof(param));
  } else {
    auto qtype = Quant::getUniformQuantizedType(input());
    dequant_fp_param_t param = {0};
    param.input_addr = Module::getAddress(input());
    param.output_addr = Module::getAddress(output());
    param.n = (int)n;
    param.c = (int)c;
    param.h = (int)h;
    param.w = (int)w;
    param.is_perchannel = false;
    param.scale_value = qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    BM1686::instance().call_global_func("backend_api_dequant_float_global",
                                        &param, sizeof(param));
  }
}

// f32
void tpu::CastOp::codegen_global_float_bm1686() {
  // same with int8
  codegen_global_int8_bm1686();
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CastOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                          int64_t out_h, int64_t out_w,
                                          int64_t out_lmem_bytes) {
  return 0;
}

void tpu::CastOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  bool qInput = Quant::isUniformQuantized(input());
  bool qOutput = Quant::isUniformQuantized(output());
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  if (!qInput && qOutput) {
    auto qtype = Quant::getUniformQuantizedType(output());
    requant_fp_param_t param = {0};
    param.input_addr = in_gi.out_addr;
    param.output_addr = gi.out_addr;
    param.requant_addr = 0;
    param.buffer_local_addr = 0;
    param.n = gi.n_slice;
    param.c = c;
    param.h = gi.h_slice;
    param.w = w;
    param.is_perchannel = false;
    param.scale_value = 1 / qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    param.output_dtype = BM168x::getDataType(output());
    param.mode = 0;
    BM1686::instance().call_local_func("backend_api_requant_float_local",
                                       &param, sizeof(param));
  } else {
    auto qtype = Quant::getUniformQuantizedType(output());
    dequant_fp_param_t param = {0};
    param.input_addr = in_gi.out_addr;
    param.output_addr = gi.out_addr;
    param.dequant_addr = 0;
    param.n = gi.n_slice;
    param.c = c;
    param.h = gi.h_slice;
    param.w = w;
    param.is_perchannel = false;
    param.scale_value = qtype.getScale();
    param.offset_value = qtype.getZeroPoint();
    param.input_dtype = BM168x::getDataType(input());
    BM1686::instance().call_local_func("backend_api_dequant_float_local",
                                       &param, sizeof(param));
  }
}
