#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  unsigned long long requant_addr;
  unsigned int buffer_local_addr;
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
  unsigned long long input_addr;
  unsigned long long output_addr;
  unsigned long long dequant_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
} dequant_fp_param_t;

void tpu::CastOp::codegen_int8_bm1686() {
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
