#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaReluOp(tpu::ReluOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_element = module::getNumElements(op.getInput());
  auto relu_limit = op.getReluLimit().convertToDouble();
  if (module::isUniformQuantized(op.getOutput())) {
    auto qtype = module::getUniformQuantizedType(op.getOutput());
    auto zero_point = qtype.getZeroPoint();
    cudaMemcpy(output, input, num_element * module::getDtypeSize(op.getOutput()),
               cudaMemcpyDeviceToDevice);
    cuda::doRelu(output, num_element, getCudaType(op.getOutput()), zero_point);
  } else if (module::getStorageType(op.getOutput()).isF32()) {
    cuda::bmActive(input, output, num_element, cuda::ACTIVE_RELU, 0, relu_limit);
  } else {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    cuda::bmActive(input_f32.get(), output_f32.get(), num_element, cuda::ACTIVE_RELU, 0, relu_limit);
    cuda::convertType(output_f32.get(), output, num_element, cuda::DT_F32, getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaReluOp(top::ReluOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_element = module::getNumElements(op.getInput());
  auto relu_limit = op.getReluLimit().convertToDouble();
  cuda::bmActive(input, output, num_element, cuda::ACTIVE_RELU, 0, relu_limit);
}