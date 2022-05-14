#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::ReshapeOp::codegen_global_int8_bm1686() {
  // do nothing
}

void tpu::ReshapeOp::codegen_local_int8_bm1686(int64_t n_step, int64_t h_step) {
  llvm_unreachable("support later");
}
