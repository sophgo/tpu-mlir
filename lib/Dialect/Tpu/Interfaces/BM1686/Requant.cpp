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
  int mul_value;
  int shift_value;
  int offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
  int reshaped_coeff;
} requant_int_param_t;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::RequantOp::codegen_global_int8_bm1686() {
  requant_int_param_t param = {0};
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  param.input_addr = Module::getAddress(input());
  param.output_addr = Module::getAddress(output());
  param.n = (int)n;
  param.c = (int)c;
  param.h = (int)h;
  param.w = (int)w;

  if (quant().getType().isa<NoneType>()) {
    param.is_perchannel = true;
    param.requant_addr = Module::getAddress(quant());
    param.reshaped_coeff = false;
  } else {
    auto qtype = Quant::getUniformQuantizedType(output());
    param.mul_value = multiplier().getValue();
    param.shift_value = rshift().getValue();
    param.offset_value = qtype.getZeroPoint();
  }
  param.mode = 2;
  BM1686::instance().call_global_func("backend_api_requant_int_global", &param,
                                      sizeof(param));
}

void tpu::RequantOp::codegen_global_float_bm1686() {
  codegen_global_int8_bm1686();
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::RequantOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                             int64_t out_h, int64_t out_w,
                                             int64_t out_lmem_bytes) {
  return 0;
}

void tpu::RequantOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}
