#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult top::PadOp::init(InferenceParameter &p) { return success(); }
void top::PadOp::deinit(InferenceParameter &p) {}

LogicalResult top::PadOp::inference(InferenceParameter &p) {
  llvm_unreachable("PadOp to be supported");
  return success();
}
