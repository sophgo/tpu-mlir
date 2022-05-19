#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

int64_t top::BatchNormOp::getFLOPs() {
  return Module::getNumElements(output()) * 2;
}

LogicalResult top::BatchNormOp::init(InferenceParameter &p) { return success(); }
void top::BatchNormOp::deinit(InferenceParameter &p) {}

LogicalResult top::BatchNormOp::inference(InferenceParameter &p) {
  llvm_unreachable("BatchNormOp to be supported");
  return success();
}
