#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult top::SoftmaxOp::init(InferenceParameter &p) { return success(); }
void top::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult top::SoftmaxOp::inference(InferenceParameter &p) {
  llvm_unreachable("SoftmaxOp to be supported");
  return success();
}
