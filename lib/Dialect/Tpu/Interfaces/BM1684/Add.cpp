
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::AddOp::codegen_global_int8_bm1684() {
  int input_num = inputs().size();
  assert(input_num == 2);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  int op_code = 1; // (0: Product; 1: Sum; 2: Max)

  auto multiplier_v = std::make_shared<std::vector<double>>(input_num, 0);
  if (coeff().hasValue()) {
    multiplier_v = Module::getF64Array(multipliers().getValue());
  }

  auto rshift_v = std::make_shared<std::vector<int64_t>>(input_num, 0);
  if (rshifts().hasValue()) {
    rshift_v = Module::getI64Array(rshifts().getValue());
  }

  BM1684::instance().dl_nodechip_eltwise_fix8b_forward_parallel(
      Module::getAddress(inputs()[0]), // u64    bottom_A_global_addr,
      Module::getAddress(inputs()[1]), // u64    bottom_B_global_addr,
      Module::getAddress(output()),    // u64    top_global_addr,
      n,                               // int    tensor_n,
      c,                               // int    tensor_c,
      h,                               // int    tensor_h,
      w,                               // int    tensor_w,
      op_code,                         // int    op_code,
      (int8_t)multiplier_v->at(0),     // int    scale_A,
      (int8_t)multiplier_v->at(1),     // int    scale_B,
      1,                               // int    sign_A,
      1,                               // int    sign_B,
      rshift_v->at(0),                 // int    rshift_A,
      rshift_v->at(1),                 // int    rshift_B,
      do_relu() ? 1 : 0,               // int    do_relu(),
      (CMD_ID_NODE *)BM1684::instance().cmdid_node // CMD_ID_NODE *id_node
  );
}

int64_t tpu::AddOp::getBufferSize_bm1684(int64_t out_n, int64_t out_c,
                                         int64_t out_h, int64_t out_w,
                                         int64_t out_lmem_bytes) {
  auto stype = Module::getStorageType(output());
  if (stype.isF32()) {
    return 0;
  }
  return out_lmem_bytes;
}

void tpu::AddOp::codegen_local_int8_bm1684(int64_t n_step, int64_t h_step) {
  auto out_ginfo = LocalGenInterface::getGroupInfo(output());
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);
  auto muls = Module::getI64Array(multipliers().getValue());
  auto rs = Module::getI64Array(rshifts().getValue());
  int num_inputs = inputs().size();
  llvm::SmallVector<int, 4> input_addrs;
  llvm::SmallVector<int, 4> input_signs;
  llvm::SmallVector<int, 4> input_strides(num_inputs * 2, 0);
  llvm::SmallVector<int, 4> mul_v(muls->begin(), muls->end());
  llvm::SmallVector<int, 4> rshift_v(rs->begin(), rs->end());
  for (int i = 0; i < num_inputs; i++) {
    auto in_ginfo = LocalGenInterface::getGroupInfo(inputs()[i]);
    input_addrs.push_back(in_ginfo.out_addr);
    auto in_type = Module::getStorageType(inputs()[i]);
    input_signs.push_back(in_type.isUnsignedInteger() == false);
  }
  BM1684::instance().dl_nodechip_eltwise_fix8b_forward_local(
      input_addrs.data(), out_ginfo.out_addr, out_ginfo.buffer_addr,
      out_ginfo.n_slice, c, out_ginfo.h_slice, w, input_strides.data(), 0, 1,
      mul_v.data(), rshift_v.data(), input_signs.data(), num_inputs, do_relu(),
      BM1684::instance().bdc_node);
}
