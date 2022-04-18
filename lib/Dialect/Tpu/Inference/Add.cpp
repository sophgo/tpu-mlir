#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::AddOp::init(InferenceParameter &p) { return success(); }
void tpu::AddOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::AddOp::inference(InferenceParameter &p) {
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
  auto dtype = output().getType().cast<RankedTensorType>().getElementType();
  auto zp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  auto b = rectified_bias();
  auto module = Module::getModuleOp(getOperation());
  auto chip = Module::getChip(module);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = 0;
    int idx = 0;
    for (auto in : p.inputs) {
      if (in != nullptr) {
        int rshift = rshifts().getValue()[idx].cast<IntegerAttr>().getInt();
        int multiplier = (int8_t)coeff()
                             .getValue()[idx]
                             .cast<FloatAttr>()
                             .getValueAsDouble();
        if (chip == Module::Chip::BM1686) {
          int tmp = in[i] * multiplier;
          int half_data = 1 << (rshift - 1);
          if (rshift > 0) {
            p.outputs[0][i] += (tmp + half_data) >> rshift;
          } else {
            p.outputs[0][i] += tmp << -rshift;
          }
        } else {
          p.outputs[0][i] += (int32_t)(in[i] * multiplier) >> rshift;
        }
      }
      idx++;
    }

    if (chip == Module::Chip::BM1686) {
      p.outputs[0][i] -=b;
      p.outputs[0][i] +=zp;
      p.outputs[0][i] = std::round(p.outputs[0][i]);
    }
    if (do_relu()) { // relu输出
      p.outputs[0][i] = p.outputs[0][i] > 255 ? 255
                        : p.outputs[0][i] < 0 ? 0
                                              : p.outputs[0][i];
    } else {
      p.outputs[0][i] = p.outputs[0][i] > 127    ? 127
                        : p.outputs[0][i] < -128 ? -128
                                                 : p.outputs[0][i];
    }
  }
#ifdef DEBUG_TPU_INFER
  llvm::errs() << "AddOp inference:" << this->name() << "\n";
  for (int i = 0; i < 5; i++) {
    printf("%d, %f+%f = %f\n", i, p.inputs[0][i], p.inputs[1][i],
    p.outputs[0][i]);
  }
#endif

  return success();
}
