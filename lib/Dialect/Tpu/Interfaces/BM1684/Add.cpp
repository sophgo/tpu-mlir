//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::AddOp::codegen_global_bm1684() {
  int input_num = getInputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  int op_code = 1; // (0: Product; 1: Sum; 2: Max)
  auto a_addr = module::getAddress(getInputs()[0]);
  auto b_addr = module::getAddress(getInputs()[1]);
  auto o_addr = module::getAddress(getOutput());
  uint64_t bottom_addrs[2] = {(uint64_t)a_addr, (uint64_t)b_addr};
  int a_shape[4] = {(int)n, (int)c, (int)h, (int)w};
  int b_shape[4] = {(int)n, (int)c, (int)h, (int)w};
  if (false == module::isUniformQuantized(getOutput())) {
    float coeff[2] = {1.0, 1.0};
    float mask[2] = {0, 0};
    BM1684::instance().dl_nodechip_eltwise_forward(
        bottom_addrs, o_addr,
        /*mask_global_offset*/ 0, /*buffer_offset*/ 0, a_shape, b_shape,
        input_num, 4, op_code, coeff, /*need_mask*/ 0, mask,
        getDoRelu() ? 1 : 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  } else {
    auto multiplier_v = module::getI64Array(getMultipliers(), input_num, 1);
    auto rshift_v = module::getI64Array(getRshifts(), input_num, 0);
    int coeff[2] = {(int)multiplier_v->at(0), (int)multiplier_v->at(1)};
    uint8_t shift[2] = {(uint8_t)rshift_v->at(0), (uint8_t)rshift_v->at(1)};
    uint8_t a_sign = module::isSign(getInputs()[0]) ? 1 : 0;
    uint8_t b_sign = module::isSign(getInputs()[1]) ? 1 : 0;
    uint8_t sign[2] = {a_sign, b_sign};
    BM1684::instance().dl_nodechip_eltwise_fix8b_forward_parallel(
        bottom_addrs, o_addr, (int)n, (int)c, (int)h, (int)w, op_code, coeff,
        sign, shift, getDoRelu() ? 1 : 0, input_num,
        (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  }
}

int64_t tpu::AddOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                         int64_t out_lmem_bytes,
                                         int64_t in_nslice, int64_t in_hslice,
                                         int64_t out_nslice,
                                         int64_t out_hslice) {
  auto stype = module::getStorageType(getOutput());
  if (stype.isF32()) {
    return 0;
  }
  return out_lmem_bytes;
}

void tpu::AddOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                      local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto out_gi = getGroupInfo(n_step, h_step);
  int num_inputs = getInputs().size();
  llvm::SmallVector<int, 8> input_addrs;
  llvm::SmallVector<int, 8> input_signs;
  llvm::SmallVector<int, 8> input_cstrides(num_inputs * 2, 0);
  for (int i = 0; i < num_inputs; i++) {
    auto in = getInputs()[i];
    auto in_ginfo = LocalGenInterface::getGroupInfo(in);
    input_addrs.push_back(in_ginfo.out_addr);
    input_signs.push_back(module::isSign(in));
  }
  if (module::isUniformQuantized(getOutput())) {
    auto muls = module::getI64Array(getMultipliers(), num_inputs, 1);
    auto rs = module::getI64Array(getRshifts(), num_inputs, 0);
    llvm::SmallVector<int, 8> mul_v(muls->begin(), muls->end());
    llvm::SmallVector<int, 8> rshift_v(rs->begin(), rs->end());
    BM1684::instance().dl_nodechip_eltwise_fix8b_forward_local(
        input_addrs.data(), out_gi.out_addr, out_gi.buffer_addr, out_gi.n_slice,
        c, out_gi.h_slice, w, input_cstrides.data(), 0, 1, mul_v.data(),
        rshift_v.data(), input_signs.data(), num_inputs, getDoRelu(),
        BM1684::instance().bdc_node);
  } else {
    llvm::SmallVector<float, 8> coeff(num_inputs, 1.0);
    BM1684::instance().dl_nodechip_eltwise_forward_local(
        input_addrs.data(), out_gi.out_addr, out_gi.n_slice, c, out_gi.h_slice,
        w, input_cstrides.data(), 0,
        /*op_code*/ 1, coeff.data(), num_inputs, getDoRelu(),
        BM1684::instance().bdc_node);
  }
}
