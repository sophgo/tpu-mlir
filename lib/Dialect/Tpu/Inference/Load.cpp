#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::LoadOp::init(InferenceParameter &p) { return success(); }
void tpu::LoadOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LoadOp::inference(InferenceParameter &p) {
  llvm_unreachable("Inference to be supported");
  return success();
}

