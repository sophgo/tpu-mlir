#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
  auto dtype = output().getType().cast<RankedTensorType>().getElementType();
  auto op = getOperation();
  auto module = Module::getModuleOp(op);
  auto chip = Module::getChip(module);
  if (dtype.isa<quant::UniformQuantizedType>()) {
    auto scale = dtype.cast<quant::UniformQuantizedType>().getScale();
    auto zp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
    llvm::errs() << "CastOp fp32 to int8 scale:" << scale << "\n";
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = p.inputs[0][i] / scale;
      if (chip == Module::Chip::BM1686 && zp) {
        p.outputs[0][i] = p.outputs[0][i] + zp;
      }
      p.outputs[0][i] = std::round(p.outputs[0][i]);
      p.outputs[0][i] = Quant::clip_to_int8(p.outputs[0][i]);
#ifdef DEBUG_TPU_INFER
      if (i < 5) printf("CastOp: %f/%f+(%ld) -> %f\n", p.inputs[0][i], scale, zp, p.outputs[0][i]);
#endif
    }
  } else if (dtype.isa<mlir::Float32Type>()) {
    auto type = input().getType().cast<RankedTensorType>();
    auto uniform_type =
        type.getElementType().cast<quant::UniformQuantizedType>();
    auto scale = uniform_type.getScale();
    llvm::errs() << "CastOp int8 to fp32 scale:" << scale << "\n";
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = scale * p.inputs[0][i];
    }
  }
  return success();
}
