//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

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
void tpu::MulOp::codegen_global_int8_bm1684x() {
  llvm_unreachable("Codegen to be supported");
}

// f32
void tpu::MulOp::codegen_global_float_bm1684x() {
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
  eltwise_float_global_param_t p = {0};
  p.input_global_addr = input_addr.data();
  p.output_global_addr = Module::getAddress(output());
  p.mask_global_addr = 0;
  p.input_num = num_inputs;
  p.n = n;
  p.c = c;
  p.h = h;
  p.w = w;
  p.op_code = ELTWISE_PRODUCT;
  p.coeff = (int *)coeffs.data();
  p.need_mask = 0;
  p.mask_index = (int *)mask_index.data();
  p.if_relu = do_relu();
  p.dtype = BM168x::getDataType(output());
  BM1684x::instance().call_global_func("backend_api_eltwise_float_global", &p,
                                       sizeof(eltwise_float_global_param_t));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::MulOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::MulOp::codegen_local_int8_bm1684x(int64_t n_step, int64_t h_step) {
  llvm_unreachable("to be supported");
}

void tpu::MulOp::codegen_local_float_bm1684x(int64_t n_step, int64_t h_step) {
  auto in0_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(inputs()[1], n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  uint32_t input_offset[] = {(uint32_t)in0_gi.out_addr,
                             (uint32_t)in1_gi.out_addr};
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto coeff_v = Module::getF64Array(coeff(), 2, 1.0);
  SmallVector<float, 2> coeff_(coeff_v->begin(), coeff_v->end());
  eltwise_float_local_param_t p = {0};
  p.input_local_addr = input_offset;
  p.buffer_local_addr = gi.buffer_addr;
  p.output_local_addr = gi.out_addr;
  p.input_num = 2;
  p.n = gi.n_slice;
  p.c = c;
  p.h = gi.h_slice;
  p.w = w;
  p.op_code = ELTWISE_PRODUCT;
  p.coeff = coeff_.data();
  p.input_local_cstride = NULL;
  p.if_relu = do_relu();
  p.dtype = BM168x::getDataType(output());
  BM1684x::instance().call_local_func("backend_api_eltwise_float_local", &p,
                                      sizeof(eltwise_float_local_param_t));
}
