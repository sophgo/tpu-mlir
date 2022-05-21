#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::ReluOp::codegen_global_int8_bm1686() {
  llvm_unreachable("Codegen to be supported");
}

void tpu::ReluOp::codegen_global_float_bm1686() {
  llvm_unreachable("Codegen to be supported");
}


// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ReluOp::getBufferSize_bm1686(int64_t out_n, int64_t out_c,
                                          int64_t out_h, int64_t out_w,
                                          int64_t out_lmem_bytes) {
  return 0;
}

void tpu::ReluOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}
