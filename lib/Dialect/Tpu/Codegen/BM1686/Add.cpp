#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

typedef struct {
  unsigned long long input_A_global_addr;
  unsigned long long input_B_global_addr;
  unsigned long long output_global_addr;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  int scale_A;
  int scale_B;
  int rshift_A;
  int rshift_B;
  int if_relu;
  DATA_TYPE_T dtype_A;
  DATA_TYPE_T dtype_B;
  int round_mode;
} eltwise_fixed_global_param_t;

typedef struct {
  unsigned int *input_local_addr;
  unsigned int output_local_addr;
  unsigned int buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  int *input_local_cstride;
  int *scale_weight;
  int *rshift;
  DATA_TYPE_T *input_dtype;
  int input_num;
  int if_relu;
  int round_mode;
} eltwise_fixed_local_param_t;

typedef struct {
  unsigned long long *input_global_addr;
  unsigned long long output_global_addr;
  unsigned long long mask_global_addr;
  int input_num;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  int *coeff;
  int need_mask;
  int *mask_index;
  int if_relu;
  DATA_TYPE_T dtype;
} eltwise_float_global_param_t;

typedef struct {
  unsigned int *input_local_addr;
  unsigned int output_local_addr;
  unsigned int buffer_local_addr;
  int input_num;
  int n;
  int c;
  int h;
  int w;
  int op_code;
  float *coeff;
  int *input_local_cstride;
  int if_relu;
  DATA_TYPE_T dtype;
} eltwise_float_local_param_t;

void tpu::AddOp::codegen_int8_bm1686() {
  eltwise_fixed_global_param_t p;
  p.input_A_global_addr = Module::getAddress(inputs()[0]);
  p.input_B_global_addr = Module::getAddress(inputs()[1]);
  p.output_global_addr = Module::getAddress(output());
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  p.n = (int)n;
  p.c = (int)c;
  p.h = (int)h;
  p.w = (int)w;
  p.op_code = 1; // (0: Product; 1: Sum; 2: Max)
  auto coeff_v = Module::getF64Array(coeff().getValue());
  auto rshift_v = Module::getI64Array(rshifts());
  p.scale_A = (int)coeff_v->at(0);
  p.scale_B = (int)coeff_v->at(1);
  p.rshift_A = (int)rshift_v->at(0);
  p.rshift_B = (int)rshift_v->at(1);
  p.if_relu = do_relu();
  p.dtype_A = BM168x::getDataType(inputs()[0]);
  p.dtype_B = BM168x::getDataType(inputs()[1]);
  p.round_mode = ROUND_UP;
  BM1686::instance().call_global_func("backend_api_eltwise_fixed_global", &p,
                                      sizeof(eltwise_fixed_global_param_t));
}
