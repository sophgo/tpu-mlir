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
typedef struct {
  uint64_t input_A_global_addr;
  uint64_t input_B_global_addr;
  uint64_t output_global_addr;
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
  uint64_t *input_global_addr;
  uint64_t output_global_addr;
  uint64_t mask_global_addr;
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
  uint32_t *input_local_addr;
  uint32_t output_local_addr;
  uint32_t buffer_local_addr;
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

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::AddOp::codegen_global_int8_bm1686() {
  int input_num = inputs().size();
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
  auto multipliers_v = Module::getI64Array(multipliers(), input_num, 1);
  auto rshift_v = Module::getI64Array(rshifts(), input_num, 0);

  p.scale_A = (int)multipliers_v->at(0);
  p.scale_B = (int)multipliers_v->at(1);
  p.rshift_A = (int)rshift_v->at(0);
  p.rshift_B = (int)rshift_v->at(1);
  p.if_relu = do_relu();
  p.dtype_A = BM168x::getDataType(inputs()[0]);
  p.dtype_B = BM168x::getDataType(inputs()[1]);
  p.round_mode = ROUND_UP;
  BM1686::instance().call_global_func("backend_api_eltwise_fixed_global", &p,
                                      sizeof(eltwise_fixed_global_param_t));
}

// f32
void tpu::AddOp::codegen_global_float_bm1686() {
  int num_inputs = inputs().size();
  llvm::SmallVector<float, 8> coeffs;
  llvm::SmallVector<float, 8> mask_index(num_inputs, 0.0f);
  llvm::SmallVector<uint64_t, 8> input_addr(num_inputs);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto coeff_v = Module::getF64Array(coeff(), num_inputs, 1.0);
  coeffs.assign(coeff_v->begin(), coeff_v->end());

  for (int i = 0; i < num_inputs; ++i) {
    mask_index[i] = i;
    input_addr[i] = Module::getAddress(inputs()[i]);
  }
  eltwise_float_global_param_t p;
  p.input_global_addr = input_addr.data();
  p.output_global_addr = Module::getAddress(output());
  p.mask_global_addr = 0;
  p.input_num = num_inputs;
  p.n = n;
  p.c = c;
  p.h = h;
  p.w = w;
  p.op_code = 1;
  p.coeff = (int *)coeffs.data();
  p.need_mask = 0;
  p.mask_index = (int *)mask_index.data();
  p.if_relu = do_relu();
  p.dtype = BM168x::getDataType(output());
  BM1686::instance().call_global_func("backend_api_eltwise_float_global", &p,
                                      sizeof(eltwise_float_global_param_t));
}

// =========================================
// LocalGenInterface
// =========================================

typedef struct {
  uint32_t *input_local_addr;
  uint32_t output_local_addr;
  uint32_t buffer_local_addr;
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

int64_t tpu::AddOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                         int64_t out_h, int64_t out_w,
                                         int64_t out_lmem_bytes) {
  auto out_type = Module::getStorageType(output());
  if (out_type.isBF16() || out_type.isF16()) {
  }
  return 0;
}

void tpu::AddOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  // int bottom_num = layer_in_tensors.size();
  // const TENSOR_PARAM_T *p_tensor_in_param =
  //     net_graph_->get_tensor_param(layer_in_tensors[0]);
  // int depth = p_tensor_in_param->d_slice;
  // DATA_TYPE_T *p_bottom_dtype = new DATA_TYPE_T[bottom_num];
  // for (int i = 0; i < bottom_num; ++i)
  //   p_bottom_dtype[i] =
  //   net_graph_->get_tensor_data_type(layer_in_tensors[i]);
  // if (p_bottom_dtype[0] == DTYPE_FP32 || p_bottom_dtype[0] == DTYPE_FP16 ||
  //     p_bottom_dtype[0] == DTYPE_BFP16) {
  //   eltwise_float_local_param_t p;
  //   p.input_local_addr = p_bottom_local_offset_;
  //   p.buffer_local_addr = imm_buffer_local_offset_;
  //   p.output_local_addr = top_local_offset_;
  //   p.input_num = bottom_num;
  //   p.n = bottom_local_dim_[0] * depth;
  //   p.c = bottom_local_dim_[1];
  //   p.h = bottom_local_dim_[2];
  //   p.w = bottom_local_dim_[3];
  //   p.op_code = layer_param->layer_param_u.eltwise_param.op_code;
  //   p.coeff = (float *)layer_param->layer_param_u.eltwise_param.bottom_coeff;
  //   p.input_local_cstride =
  //       bottom_chstride_en_ ? p_bottom_local_chstride_ : NULL;
  //   p.if_relu = layer_param->if_relu ? 1 : 0;
  //   p.dtype = p_bottom_dtype[0];
  //   call_local_func("backend_api_eltwise_float_local", &p,
  //                   sizeof(eltwise_float_local_param_t));
  // } else if (p_bottom_dtype[0] == DTYPE_INT8 ||
  //            p_bottom_dtype[0] == DTYPE_UINT8) {
  //   eltwise_fixed_local_param_t p;
  //   p.output_local_addr = top_local_offset_;
  //   p.input_local_addr = p_bottom_local_offset_;
  //   p.buffer_local_addr = imm_buffer_local_offset_;
  //   p.n = bottom_local_dim_[0];
  //   p.c = bottom_local_dim_[1] * depth;
  //   p.h = bottom_local_dim_[2];
  //   p.w = bottom_local_dim_[3];
  //   p.op_code = layer_param->layer_param_u.eltwise_param.op_code;
  //   p.input_local_cstride =
  //       bottom_chstride_en_ ? p_bottom_local_chstride_ : NULL;
  //   p.scale_weight =
  //       (int *)layer_param->layer_param_u.eltwise_param.bottom_coeff;
  //   p.rshift = (int *)layer_param->layer_param_u.eltwise_param.in_rshift_num;
  //   p.input_dtype = p_bottom_dtype;
  //   p.input_num = bottom_num;
  //   p.if_relu = layer_param->if_relu ? 1 : 0;
  //   p.round_mode = ROUND_UP;
  //   call_local_func("backend_api_eltwise_fixed_local", &p,
  //                   sizeof(eltwise_fixed_local_param_t));
  // }
  // delete[] p_bottom_dtype;
}
