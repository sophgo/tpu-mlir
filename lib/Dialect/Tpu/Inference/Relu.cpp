#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::ReluOp::inference(InferenceParameter &p) {
  relu(p.inputs[0], p.outputs[0], Module::getNumElements(output()),
       Module::getStorageType(output()));
  return success();
}

