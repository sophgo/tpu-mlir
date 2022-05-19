#include "sophgo/Backend/BM168x/BM1684.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/MathUtils.h"

using namespace mlir;
using namespace sophgo;
using namespace sophgo::helper;
using namespace sophgo::backend;

void tpu::SoftmaxOp::codegen_global_int8_bm1684() {
  llvm_unreachable("Codegen to be supported");
}

