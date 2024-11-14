#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RopeOp::getFLOPs() {
  return module::getNumElements(getInput1()) * 4;
}

LogicalResult top::RopeOp::init(InferenceParameter &p) {

  auto binary = new Binary();
  p.handle = (void *)binary;
  return success();
}

void top::RopeOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult top::RopeOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput1());
  float *temp_input = new float[num_element];
  float *temp_result0 = new float[num_element];
  float *temp_result1 = new float[num_element];
  auto input_shape = module::getShape(getInput1());
  auto weight_shape = module::getShape(getInput2());
  auto binary = (Binary *)p.handle;

#pragma omp parallel for schedule(static, omp_schedule(num_element))

  for (int i = 0; i < num_element; i++) {
    temp_input[i] = p.inputs[0][i];
  }

  for (int i = 0; i < num_element; i++) {
    if (i % 2 == 0 && i + 1 < num_element) {
      float temp = temp_input[i];
      temp_input[i] = temp_input[i + 1];
      temp_input[i + 1] = temp;
      temp_input[i] = -temp_input[i];
    }
  }

  float *weight0 = p.inputs[1];
  float *weight1 = p.inputs[2];
  float *input = p.inputs[0];

  (*binary)
      .hs(temp_input, weight0, input_shape, weight_shape)
      .dst(temp_result0, module::getShape(getOutput()))
      .algorithem(algorithm::binary_mul)
      .setup();
  binary->run();

  (*binary)
      .hs(input, weight1, input_shape, weight_shape)
      .dst(temp_result1, module::getShape(getOutput()))
      .algorithem(algorithm::binary_mul)
      .setup();
  binary->run();

  for (int i = 0; i < num_element; i++) {
    p.outputs[0][i] = temp_result0[i] + temp_result1[i];
  }

  delete[] temp_input;
  delete[] temp_result0;
  delete[] temp_result1;

  return success();
}

void top::RopeOp::shape_inference() { common_shape_inference(getOperation()); }