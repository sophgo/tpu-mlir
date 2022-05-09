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
  auto scale = dtype.cast<quant::UniformQuantizedType>().getScale();
  auto b = rectified_bias();
  auto module = Module::getModuleOp(getOperation());
  auto chip = Module::getChip(module);

  int idx = 0;
  int alg_type = 0; // 0:normal, 1:only rightshift the output, 2:use fp32
                    // compute
  if (chip == Module::Chip::BM1686) {
    alg_type = 2;
  }
  std::vector<std::string> input_names;
  std::vector<float *> inputs;
  if (alg_type == 2) {
    auto op = getOperation();
    for (int i = 0; i < p.inputs.size(); i++) {
      auto tmp_buf = std::make_shared<std::vector<float>>(num_elem);
      inputs.push_back(tmp_buf->data());
      auto input = op->getOperand(i);
      if (Quant::isUniformQuantized(input)) {
        auto qtype = Quant::getUniformQuantizedType(input);
        for (int64_t j = 0; j < num_elem; j++) {
          tmp_buf->data()[j] =
              (p.inputs[i][j] - qtype.getZeroPoint()) * qtype.getScale();
        }
      }
    }
  } else {
    inputs.assign(p.inputs.begin(), p.inputs.end());
  }

#ifdef DEBUG_TPU_INFER
  printf("inference rectified_bias:%f\n", b.convertToDouble());
  auto op = getOperation();
  for (int i = 0; i < p.inputs.size(); i++) {
    auto input = op->getOperand(i);
    if (Quant::isUniformQuantized(input)) {
      auto qtype = Quant::getUniformQuantizedType(input);
      printf("i:%d, zp:%d, scale:%f\n", i, qtype.getZeroPoint(),
             qtype.getScale());
    }
    int rshift = rshifts().getValue()[i].cast<IntegerAttr>().getInt();
    int multiplier = multipliers().getValue()[i].cast<IntegerAttr>().getInt();
    printf("i:%d, multiplier:%d, rshift:%d, rectified_bias:%f\n", i, multiplier,
           rshift, b.convertToDouble());
  }
#endif

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = 0;
    int idx = 0;
    for (auto in : inputs) {
      if (in != nullptr) {
        int64_t rshift = rshifts().getValue()[idx].cast<IntegerAttr>().getInt();
        int64_t multiplier =
            multipliers().getValue()[idx].cast<IntegerAttr>().getInt();
        if (chip == Module::Chip::BM1686) {
          if (alg_type == 1) {
            p.outputs[0][i] += ((int64_t)in[i]) * multiplier;
          } else if (alg_type == 2) {
            p.outputs[0][i] += in[i];
          } else {
            int64_t tmp = ((int64_t)in[i]) * multiplier;
            if (rshift > 0) {
              int half_data = 1 << (rshift - 1);
              p.outputs[0][i] += (tmp + half_data) >> rshift;
            } else {
              p.outputs[0][i] += tmp << -rshift;
            }
          }
        } else {
          p.outputs[0][i] += (int32_t)(in[i] * multiplier) >> rshift;
        }
      }
      idx++;
    }

    if (chip == Module::Chip::BM1686) {
      if (alg_type == 1) {
        p.outputs[0][i] -= b.convertToDouble();
        int64_t rshift = rshifts().getValue()[0].cast<IntegerAttr>().getInt();
        int64_t tmp = std::round(p.outputs[0][i]);
        p.outputs[0][i] = tmp >> rshift;
      } else if (alg_type != 2) {
        p.outputs[0][i] -= b.convertToDouble();
      }
    }

    if (do_relu()) {
      p.outputs[0][i] = p.outputs[0][i] > 0 ? p.outputs[0][i] : 0;
    }

    if (chip == Module::Chip::BM1686) {
      if (alg_type == 2) {
        p.outputs[0][i] = std::round(p.outputs[0][i] / scale) + zp;
      } else {
        p.outputs[0][i] += zp;
        p.outputs[0][i] = std::round(p.outputs[0][i]);
      }
    }

    if (do_relu()) { // relu输出
      if (chip == Module::Chip::BM1686) {
        p.outputs[0][i] = Quant::clip_to_int8(p.outputs[0][i]);
      } else {
        p.outputs[0][i] = Quant::clip_to_uint8(
            p.outputs[0][i]); // 1684量化这里要设为uint8才能过 todo
      }
    } else {
      p.outputs[0][i] = Quant::clip_to_int8(p.outputs[0][i]);
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
