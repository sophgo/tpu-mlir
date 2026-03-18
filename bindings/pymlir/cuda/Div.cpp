#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaDivOp(tpu::DivOp op) {
  auto out = op.getOutput();
  auto num_inputs = op.getInputs().size();
  if (num_inputs != 2) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto in0 = op.getInputs()[0];
  auto in1 = op.getInputs()[1];
  auto input0 = getCudaData(in0);
  auto input1 = getCudaData(in1);
  auto output = getCudaData(out);
  auto shape0 = std::vector<int64_t>(module::getShape(in0));
  auto shape1 = std::vector<int64_t>(module::getShape(in1));
  auto shape2 = std::vector<int64_t>(module::getShape(out));
  auto is_reverse = op.getIsReverse();
  auto num_dims = shape0.size();
  auto do_relu = op.getDoRelu();
  if (module::isUniformQuantized(op.getOutput())) {
    llvm_unreachable("Not Implemented");
  } else if (module::getStorageType(op.getOutput()).isF32()) {
    if (is_reverse) {
      cuda::divMDF32(input1, input0, output, shape1.data(), shape0.data(), shape2.data(), num_dims);
    } else {
      cuda::divMDF32(input0, input1, output, shape0.data(), shape1.data(), shape2.data(), num_dims);
    }
    if (do_relu) {
      auto relu_limit = op.getReluLimit().convertToDouble();
      cuda::bmActive(output, output, module::getNumElements(out),
                    cuda::ACTIVE_RELU, 0, relu_limit);
    }
  } else {
    if (module::getStorageType(op.getOutput()).isFloat8E4M3FN()) {
      llvm_unreachable("Not Implemented");
    }
    auto input0_f32 = newCudaData(in0, cuda::DT_F32);
    auto input1_f32 = newCudaData(in1, cuda::DT_F32);
    auto output_f32 = newCudaData(out, cuda::DT_F32);
    if (is_reverse) {
      cuda::divMDF32(input1_f32.get(), input0_f32.get(), output_f32.get(), shape1.data(), shape0.data(), shape2.data(), num_dims);
    } else {
      cuda::divMDF32(input0_f32.get(), input1_f32.get(), output_f32.get(), shape0.data(), shape1.data(), shape2.data(), num_dims);
    }
    if (do_relu) {
      auto relu_limit = op.getReluLimit().convertToDouble();
      cuda::bmActive(output_f32.get(), output_f32.get(), module::getNumElements(out),
                    cuda::ACTIVE_RELU, 0, relu_limit);
    }
    cuda::convertType(output_f32.get(), output, module::getNumElements(out), cuda::DT_F32, getCudaType(out));
    input0_f32.reset();
    input1_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaDivOp(top::DivOp op) {
  auto out = op.getOutput();
  auto num_inputs = op.getInputs().size();
  if (2 != num_inputs) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto in0 = op.getInputs()[0];
  auto in1 = op.getInputs()[1];
  auto input0 = getCudaData(in0);
  auto input1 = getCudaData(in1);
  auto output = getCudaData(out);
  auto shape0 = std::vector<int64_t>(module::getShape(in0));
  auto shape1 = std::vector<int64_t>(module::getShape(in1));
  auto shape2 = std::vector<int64_t>(module::getShape(out));
  auto is_reverse = op.getIsReverse();
  auto num_dims = shape0.size();
  auto do_relu = op.getDoRelu();
  if (is_reverse) {
    cuda::divMDF32(input1, input0, output, shape1.data(), shape0.data(), shape2.data(), num_dims);
  } else {
    cuda::divMDF32(input0, input1, output, shape0.data(), shape1.data(), shape2.data(), num_dims);
  }
  if (do_relu) {
    auto relu_limit = op.getReluLimit().convertToDouble();
    cuda::bmActive(output, output, module::getNumElements(out),
                   cuda::ACTIVE_RELU, 0, relu_limit);
  }
}