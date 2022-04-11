#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult top::ReluOp::inference(InferenceParameter &p) {
  relu(p.inputs[0], p.outputs[0], Module::getNumElements(input()));
  return success();
}
