#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

int64_t top::SoftmaxOp::getFLOPs() {
  //   2*n          -- compute shifted logits
  //   n            -- exp of shifted logits
  //   2*n          -- compute softmax from exp of shifted logits
  return Module::getNumElements(input()) * 5;
}

LogicalResult top::SoftmaxOp::init(InferenceParameter &p) { return success(); }
void top::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult top::SoftmaxOp::inference(InferenceParameter &p) {
  llvm_unreachable("SoftmaxOp to be supported");
  return success();
}
