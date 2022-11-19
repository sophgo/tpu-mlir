#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
// #include "tpu_mlir/Backend/BM168x/cv18xx.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
// using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::SigmoidOp::codegen_global_cv18xx( int64_t layer_id) {
  llvm_unreachable("Not supported now");
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SigmoidOp::getBufferSize_cv18xx(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  llvm_unreachable("Not supported now");
  return 0;
}

void tpu::SigmoidOp::codegen_local_cv18xx(int64_t n_step, int64_t h_step) {
  llvm_unreachable("Not supported now");
}
