#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaMinConstOp(tpu::MinConstOp op) {
  auto do_relu = op.getDoRelu();
  auto const_val = op.getConstVal().convertToDouble();
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_element = module::getNumElements(op.getInput());
  auto relu_limit = op.getReluLimit().convertToDouble();
  if (module::isUniformQuantized(op.getOutput())) {
    auto in_type = module::getUniformQuantizedType(op.getInput());
    auto out_type = module::getUniformQuantizedType(op.getOutput());
    auto input_signed = !module::getStorageType(op.getInput()).isUnsignedInteger();
    auto output_signed = !module::getStorageType(op.getOutput()).isUnsignedInteger();
    auto input_zp = in_type.getZeroPoint();
    auto output_zp = out_type.getZeroPoint();
    cuda::minConstI8(input, const_val, output, op.getMultiplier(), op.getRshift(),
                    input_zp, output_zp, num_element, input_signed, output_signed,
                    do_relu);
  } else if (module::getStorageType(op.getOutput()).isF32()) {
    if (do_relu && relu_limit > 0.0) const_val = std::min(const_val, relu_limit);
    double clip_min = do_relu ? 0.0 : -1e30;
    cuda::clip(input, output, num_element, clip_min, const_val);
  } else {
    if (module::getStorageType(op.getOutput()).isFloat8E4M3FN()) {
      llvm_unreachable("Not Implemented");
    }
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    if (do_relu && relu_limit > 0.0) const_val = std::min(const_val, relu_limit);
    double clip_min = do_relu ? 0.0 : -1e30;
    cuda::clip(input_f32.get(), output_f32.get(), num_element, clip_min, const_val);
    cuda::convertType(output_f32.get(), output, num_element, cuda::DT_F32, getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaMinConstOp(top::MinConstOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_element = module::getNumElements(op.getInput());
  auto const_val = op.getConstVal().convertToDouble();
  cuda::clip(input, output, num_element, -1e30, const_val);
}