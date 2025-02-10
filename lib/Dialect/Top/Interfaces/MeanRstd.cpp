#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::MeanRstdOp::getFLOPs() {
  return module::getNumElements(getMean()) * 2;
}

LogicalResult top::MeanRstdOp::init(InferenceParameter &p) { return success(); }
void top::MeanRstdOp::deinit(InferenceParameter &p) {}

LogicalResult top::MeanRstdOp::inference(InferenceParameter &p) {
  float *input = p.inputs[0];
  float *running_mean = p.inputs[1];
  float *running_var = p.inputs[2];
  float *gamma = p.inputs[3];
  float *beta = p.inputs[4];
  float *saved_mean = p.outputs[0];
  float *saved_rstd = p.outputs[1];
  float *running_mean_update = p.outputs[2];
  float *running_var_update = p.outputs[3];
  float *scale = p.outputs[4];
  float *bias = p.outputs[5];
  auto input_shape = module::getShape(getInput());
  auto eps = getEps().convertToDouble();
  auto momentum = getMomentum().convertToDouble();
  auto input_n = input_shape[0];
  auto input_c = input_shape[1];
  auto input_h = input_shape[2];
  auto input_w = input_shape[3];
  for (int i_c = 0; i_c < input_c; i_c++) {
    float sum = 0;
    for (int i_n = 0; i_n < input_n; i_n++) {
      for (int i_h = 0; i_h < input_h; i_h++) {
        for (int i_w = 0; i_w < input_w; i_w++) {
          int pos = i_n * input_c * input_h * input_w +
                    i_c * input_h * input_w + i_h * input_w + i_w;
          sum += input[pos];
        }
      }
    }
    saved_mean[i_c] = sum / (input_n * input_h * input_w);
  }
  float *var = new float[input_c];
  for (int i_c = 0; i_c < input_c; i_c++) {
    float sum_tmp = 0;
    for (int i_n = 0; i_n < input_n; i_n++) {
      for (int i_h = 0; i_h < input_h; i_h++) {
        for (int i_w = 0; i_w < input_w; i_w++) {
          int pos = i_n * input_c * input_h * input_w +
                    i_c * input_h * input_w + i_h * input_w + i_w;
          sum_tmp +=
              (input[pos] - saved_mean[i_c]) * (input[pos] - saved_mean[i_c]);
        }
      }
    }
    var[i_c] = sum_tmp / (input_n * input_h * input_w - 1);
    saved_rstd[i_c] =
        1.0 / std::sqrt((sum_tmp / (input_n * input_h * input_w)) + eps);
  }
  for (int i_c = 0; i_c < input_c; i_c++) {
    running_mean_update[i_c] =
        momentum * saved_mean[i_c] + (1 - momentum) * running_mean[i_c];
    running_var_update[i_c] =
        momentum * var[i_c] + (1 - momentum) * running_var[i_c];
    scale[i_c] = gamma[i_c] * saved_rstd[i_c];
    bias[i_c] = beta[i_c] - saved_mean[i_c] * scale[i_c];
  }
  delete[] var;
  return success();
}

void top::MeanRstdOp::shape_inference() {
  // auto input_shape        = module::getShape(getInput());
  auto running_mean_shape = module::getShape(getRunningMean());
  auto running_var_shape = module::getShape(getRunningVar());
  auto weight_shape = module::getShape(getWeight());
  auto bias_shape = module::getShape(getBias());
  long weight_new_shape[4] = {1, weight_shape[1], 1, 1};
  long bias_new_shape[4] = {1, bias_shape[1], 1, 1};
  module::setShapeOrVerify(getMean(), running_mean_shape);
  module::setShapeOrVerify(getRstd(), running_var_shape);
  module::setShapeOrVerify(getRunningMeanUpdate(), running_mean_shape);
  module::setShapeOrVerify(getRunningVarUpdate(), running_var_shape);
  module::setShapeOrVerify(getScale(), weight_new_shape);
  module::setShapeOrVerify(getBiasNew(), bias_new_shape);
}
