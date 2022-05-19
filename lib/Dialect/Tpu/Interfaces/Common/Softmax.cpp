#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::SoftmaxOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  auto dt = getDnnlType(input());

  int izp = 0;
  auto dtype = input().getType().cast<RankedTensorType>().getElementType();
  if (dtype.isa<quant::UniformQuantizedType>()) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }
  p.handle = (void *)pooling;
  return success();
}

void tpu::SoftmaxOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::SoftmaxOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();
  if (do_relu()) {
    relu(p.outputs[0], p.outputs[0], Module::getNumElements(output()),
         Module::getStorageType(output()));
  }
  return success();
}
