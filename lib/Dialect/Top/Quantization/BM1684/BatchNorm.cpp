#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

Value top::BatchNormOp::quantize_int8_bm1684() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}
