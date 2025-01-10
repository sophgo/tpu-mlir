//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::BatchNormTrainOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 6;
}

LogicalResult top::BatchNormTrainOp::init(InferenceParameter &p) {
  return success();
}
void top::BatchNormTrainOp::deinit(InferenceParameter &p) {}

LogicalResult top::BatchNormTrainOp::inference(InferenceParameter &p) {
  const float momentum = getMomentum().convertToDouble();
  const float eps = getEpsilon().convertToDouble();
  const auto input_shape = module::getShape(getInput());

  const float *input_data = p.inputs[0];
  const float *mean_data = p.inputs[1];
  const float *var_data = p.inputs[2];
  const float *gamma_data = p.inputs[3];
  const float *beta_data = p.inputs[4];

  bool do_relu = getDoRelu();

  float *output_data = p.outputs[0];
  float *saved_mean_data = p.outputs[1];
  float *saved_invstd = p.outputs[2];
  float *running_mean = p.outputs[3];
  float *running_var = p.outputs[4];

  float *var_temp = new float[input_shape[1]];

  int n = input_shape[0];
  int c = input_shape[1];
  int h = input_shape[2];
  int w = input_shape[3];
  for (int ci = 0; ci < c; ci++) {
    float cur_mean = 0;
    for (int ni = 0; ni < n; ni++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int index = ni * c * h * w + ci * h * w + hi * w + wi;
          cur_mean += input_data[index];
        }
      }
    }
    cur_mean /= n * h * w;
    saved_mean_data[ci] = cur_mean;
  }

  for (int ci = 0; ci < c; ci++) {
    float cur_var = 0;
    float cur_save_mean = saved_mean_data[ci];
    for (int ni = 0; ni < n; ni++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int index = ni * c * h * w + ci * h * w + hi * w + wi;
          cur_var += (input_data[index] - cur_save_mean) *
                     (input_data[index] - cur_save_mean);
        }
      }
    }
    cur_var /= n * h * w;
    var_temp[ci] = cur_var;
    saved_invstd[ci] = 1 / std::sqrt(cur_var + eps);
  }

  // running mean var
  for (int ci = 0; ci < c; ci++) {
    running_mean[ci] =
        (1 - momentum) * mean_data[ci] + momentum * saved_mean_data[ci];
    running_var[ci] = (1 - momentum) * var_data[ci] + momentum * var_temp[ci];
  }

  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          int index = ni * c * h * w + ci * h * w + hi * w + wi;
          output_data[index] =
              (input_data[index] - saved_mean_data[ci]) * saved_invstd[ci];
          output_data[index] *= gamma_data[ci];
          output_data[index] += beta_data[ci];
          if (do_relu) {
            output_data[index] =
                output_data[index] > 0 ? output_data[index] : 0;
          }
        }
      }
    }
  }
  delete[] var_temp;
  return success();
}

void top::BatchNormTrainOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto mean_shape = module::getShape(getMean());
  module::setShapeOrVerify(getOutput(), input_shape);
  module::setShapeOrVerify(getMeanOut(), {1, mean_shape[0], 1, 1});
  module::setShapeOrVerify(getSavedInvstd(), {1, mean_shape[0], 1, 1});
  module::setShapeOrVerify(getRunningMean(), {1, mean_shape[0], 1, 1});
  module::setShapeOrVerify(getRunningVar(), {1, mean_shape[0], 1, 1});
}
