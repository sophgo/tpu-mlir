#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::SoftmaxOp::codegen_global_int8_bm1686() {
  llvm_unreachable("Codegen to be supported");
}

void tpu::SoftmaxOp::codegen_global_float_bm1686() {
  llvm_unreachable("Codegen to be supported");
}
