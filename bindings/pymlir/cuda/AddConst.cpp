#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaAddConstOp(tpu::AddConstOp op) {
  bool do_relu = op.getDoRelu();
  const float const_val = op.getConstVal().convertToDouble();
  auto num_elem = module::getNumElements(op.getOutput());
  if (module::isUniformQuantized(op.getOutput())) {
    auto input_qtype = module::getUniformQuantizedType(op.getInput());
    auto output_qtype = module::getUniformQuantizedType(op.getOutput());
    auto input_zp = input_qtype.getZeroPoint();
    auto output_zp = output_qtype.getZeroPoint();
    auto input_signed = !module::getStorageType(op.getInput()).isUnsignedInteger();
    auto output_signed = !module::getStorageType(op.getOutput()).isUnsignedInteger();
    cuda::addConstI8(getCudaData(op.getInput()), const_val, getCudaData(op.getOutput()),
                    op.getMultiplier(), op.getRshift(), input_zp, output_zp, num_elem,
                    input_signed, output_signed, do_relu);
  } else if (module::getStorageType(op.getInput()).isF32()) {
    void *input = getCudaData(op.getInput());
    void *output = getCudaData(op.getOutput());
    cuda::subConst4DF32(input, -const_val, output, do_relu, false, 1, 1, 1, num_elem);
  } else {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    if (module::getStorageType(op.getInput()).isFloat8E4M3FN()) {
      double scale = op.getF8Scale().convertToDouble();
      cuda::mulConst6DF32(input_f32.get(), scale, input_f32.get(), false,
                          1, 1, 1, num_elem, 1, 1);
    }
    cuda::subConst4DF32(input_f32.get(), -const_val, output_f32.get(), do_relu,
                        false, 1, 1, 1, num_elem);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), num_elem,
                      cuda::DT_F32, getCudaType(op.getOutput()));
    input_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaAddConstOp(top::AddConstOp op) {
  bool do_relu = op.getDoRelu();
  const float const_val = op.getConstVal().convertToDouble();
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_elem = module::getNumElements(op.getOutput());
  cuda::subConst4DF32(input, -const_val, output, do_relu, false, 1, 1, 1, num_elem);
}