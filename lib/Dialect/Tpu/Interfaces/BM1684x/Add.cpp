//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "Binary_param.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::AddOp::codegen_global_bm1684x() {
  int num_inputs = inputs().size();
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto out_type = Module::getStorageType(output());
  auto in_type = Module::getStorageType(inputs()[0]);
  if (Quant::isUniformQuantized(output())) {
    eltwise_fixed_global_param_t p;
    p.input_A_global_addr = Module::getAddress(inputs()[0]);
    p.input_B_global_addr = Module::getAddress(inputs()[1]);
    p.output_global_addr = Module::getAddress(output());
    p.n = (int)n;
    p.c = (int)c;
    p.h = (int)h;
    p.w = (int)w;
    p.op_code = ELTWISE_ADD;
    auto multipliers_v = Module::getI64Array(multipliers(), num_inputs, 1);
    auto rshift_v = Module::getI64Array(rshifts(), num_inputs, 0);
    p.scale_A = (int)multipliers_v->at(0);
    p.scale_B = (int)multipliers_v->at(1);
    p.rshift_A = (int)rshift_v->at(0);
    p.rshift_B = (int)rshift_v->at(1);
    p.if_relu = do_relu();
    p.dtype_A = BM168x::getDataType(inputs()[0]);
    p.dtype_B = BM168x::getDataType(inputs()[1]);
    p.round_mode = ROUND_UP;
    BM1684x::instance().call_global_func("backend_api_eltwise_fixed_global", &p,
                                         sizeof(eltwise_fixed_global_param_t));
  } else if (in_type.isInteger(32) && out_type.isInteger(32)) {
    auto op = getOperation();
    auto input_spec = BM1684x::get_input_spec(op);
    auto output_spec = BM1684x::get_output_spec(op);

    bcbinary_common_spec_t param = {0};
    param.binary_type = BINARY_ADD;
    param.if_relu = do_relu();
    param.relu_upper_limit = relu_limit().convertToDouble();
    param.rshift_A = 0;
    param.rshift_B = 0;
    param.scale_A = 1;
    param.scale_B = 1;
    BM1684x::instance().call_global_func("backend_api_bcbinary_global", &param,
                                        sizeof(param), input_spec->data(),
                                        output_spec->data());
  } else {
    llvm::SmallVector<float, 8> coeffs;
    llvm::SmallVector<float, 8> mask_index(num_inputs, 0.0f);
    llvm::SmallVector<uint64_t, 8> input_addr(num_inputs);
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
    p.op_code = ELTWISE_ADD;
    p.coeff = (int *)coeffs.data();
    p.need_mask = 0;
    p.mask_index = (int *)mask_index.data();
    p.if_relu = do_relu();
    p.dtype = BM168x::getDataType(output());
    BM1684x::instance().call_global_func("backend_api_eltwise_float_global", &p,
                                         sizeof(eltwise_float_global_param_t));
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::AddOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  auto out_type = Module::getStorageType(output());
  if (out_type.isInteger(8)) {
    // INT16 as middle result
    return 2 * out_lmem_bytes * sizeof(int16_t);
  } else if (out_type.isBF16() || out_type.isF16()) {
    return out_lmem_bytes;
  }
  return 0;
}

void tpu::AddOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  auto in0_gi = LocalGenInterface::getGroupInfo(inputs()[0], n_step, h_step);
  auto in1_gi = LocalGenInterface::getGroupInfo(inputs()[1], n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  uint32_t input_offset[] = {(uint32_t)in0_gi.out_addr,
                             (uint32_t)in1_gi.out_addr};
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto out_type = Module::getStorageType(output());
  auto in_type = Module::getStorageType(inputs()[0]);
  if (Quant::isUniformQuantized(inputs()[0], output())) {
    auto multiplier_v = Module::getI64Array(multipliers(), 2, 1);
    auto rshift_v = Module::getI64Array(rshifts(), 2, 0);
    SmallVector<int32_t, 2> multi_v(multiplier_v->begin(), multiplier_v->end());
    SmallVector<int32_t, 2> r_v(rshift_v->begin(), rshift_v->end());
    DATA_TYPE_T input_types[2] = {BM1684x::getDataType(inputs()[0]),
                                  BM1684x::getDataType(inputs()[1])};
    eltwise_fixed_local_param_t p = {0};
    p.input_local_addr = input_offset;
    p.buffer_local_addr = gi.buffer_addr;
    p.output_local_addr = gi.out_addr;
    p.input_num = 2;
    p.input_dtype = input_types;
    p.input_local_cstride = nullptr;
    p.n = gi.n_slice;
    p.c = c;
    p.h = gi.h_slice;
    p.w = w;
    p.op_code = ELTWISE_ADD;
    p.scale_weight = multi_v.data();
    p.rshift = r_v.data();
    p.if_relu = do_relu();
    p.round_mode = ROUND_UP;
    BM1684x::instance().call_local_func("backend_api_eltwise_fixed_local", &p,
                                        sizeof(eltwise_fixed_local_param_t));
  } else if (in_type.isInteger(32) && out_type.isInteger(32)) {
    auto op = getOperation();
    auto input_spec = BM1684x::get_input_spec(op);
    auto output_spec = BM1684x::get_output_spec(op);
    local_sec_info_t sec_info;
    memset(&sec_info, 0, sizeof(sec_info));
    sec_info.n_slice = gi.n_slice;
    sec_info.out_n_slice = gi.n_slice;

    sec_info.is_h_split = !(gi.h_idx == 0 && gi.h_slice == h);
    sec_info.h_idx = in0_gi.h_idx;
    sec_info.h_slice = in0_gi.h_slice;
    sec_info.out_h_idx = gi.h_idx;
    sec_info.out_h_slice = gi.h_slice;

    sec_info.is_w_split = false;
    sec_info.w_slice = w;
    sec_info.out_w_slice = w;

    bcbinary_local_param_t param = {0};
    param.spec.common.binary_type = BINARY_ADD;
    param.spec.common.if_relu = do_relu();
    param.spec.common.relu_upper_limit = relu_limit().convertToDouble();
    param.spec.common.rshift_A = 0;
    param.spec.common.rshift_B = 0;
    param.spec.common.scale_A = 1;
    param.spec.common.scale_B = 1;
    param.spec.buffer_addr = gi.buffer_addr;
    param.A_is_coeff = false;
    param.B_is_coeff = false;
    BM1684x::instance().call_local_func("backend_api_bcbinary_local", &param,
                                        sizeof(param), &sec_info, input_spec->data(),
                                        output_spec->data());
  } else {
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
    p.op_code = ELTWISE_ADD;
    p.coeff = coeff_.data();
    p.input_local_cstride = NULL;
    p.if_relu = do_relu();
    p.dtype = BM168x::getDataType(output());
    BM1684x::instance().call_local_func("backend_api_eltwise_float_local", &p,
                                        sizeof(eltwise_float_local_param_t));
  }
}
