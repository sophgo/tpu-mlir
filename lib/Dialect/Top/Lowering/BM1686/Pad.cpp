#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

Value top::PadOp::lowering_int8_bm1686() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}

Value top::PadOp::lowering_f32_bm1686() {
  llvm_unreachable("BatchNormOp to be supported");
  return nullptr;
}
