#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::AddOp::inference(InferenceParameter &p) {
  auto num_elem = output().getType().cast<RankedTensorType>().getNumElements();
  auto dtype = output().getType().cast<RankedTensorType>().getElementType();
  auto zp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
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
        p.outputs[0][i] += (int32_t)(in[i] * multiplier) >> rshift;
      }
      idx++;
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
  // llvm::errs() << "AddOp inference:" << this->name() << "\n";
  for (int i = 0; i < 5; i++) {
    // printf("%d, %f+%f = %f\n", i, p.inputs[0][i], p.inputs[1][i],
    // p.outputs[0][i]);
  }
  return success();
}

