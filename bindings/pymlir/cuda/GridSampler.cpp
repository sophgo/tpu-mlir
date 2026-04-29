#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaGridSamplerOp(tpu::GridSamplerOp op) {
  cuda::grid_sample_interpolation_mode_t interpolation_mode =
    static_cast<cuda::grid_sample_interpolation_mode_t>(op.getMode());
  cuda::grid_sample_padding_mode_t padding_mode =
    static_cast<cuda::grid_sample_padding_mode_t>(op.getPaddingMode());
  auto align_corners = op.getAlignCorners();
  void *input = getCudaData(op.getInput());
  void *grid = getCudaData(op.getGrid());
  void *output = getCudaData(op.getOutput());
  auto input_shape = module::getShape(op.getInput());
  auto grid_shape = module::getShape(op.getGrid());
  if (input_shape.size() != 4) {
    llvm_unreachable("Only support 4D input for GridSampler now");
  }
  if (module::isUniformQuantized(op.getOutput())) {
    llvm_unreachable("Not support quantized output for GridSampler now");
  } else if (module::getStorageType(op.getOutput()).isF32()) {
    cuda::GridSample4D(input, grid, output, input_shape[0], input_shape[1],
             input_shape[2], input_shape[3], grid_shape[1],
             grid_shape[2], align_corners, interpolation_mode,
             padding_mode);
  } else {
    if (module::getStorageType(op.getOutput()).isFloat8E4M3FN()) {
      llvm_unreachable("Not Implemented");
    }
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto grid_f32 = newCudaData(op.getGrid(), cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    cuda::GridSample4D(input_f32.get(), grid_f32.get(), output_f32.get(),
             input_shape[0], input_shape[1], input_shape[2], input_shape[3],
             grid_shape[1], grid_shape[2], align_corners, interpolation_mode,
             padding_mode);
    cuda::convertType(output_f32.get(), output, module::getNumElements(op.getOutput()),
                      cuda::DT_F32, getCudaType(op.getOutput()));
    input_f32.reset();
    grid_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaGridSamplerOp(top::GridSamplerOp op) {
  cuda::grid_sample_interpolation_mode_t interpolation_mode =
    static_cast<cuda::grid_sample_interpolation_mode_t>(op.getMode());
  cuda::grid_sample_padding_mode_t padding_mode =
    static_cast<cuda::grid_sample_padding_mode_t>(op.getPaddingMode());
  auto align_corners = op.getAlignCorners();
  void *input = getCudaData(op.getInput());
  void *grid = getCudaData(op.getGrid());
  void *output = getCudaData(op.getOutput());
  auto input_shape = module::getShape(op.getInput());
  auto grid_shape = module::getShape(op.getGrid());
  if (input_shape.size() != 4) {
    llvm_unreachable("Only support 4D input for GridSampler now");
  }
  cuda::GridSample4D(input, grid, output, input_shape[0], input_shape[1],
             input_shape[2], input_shape[3], grid_shape[1],
             grid_shape[2], align_corners, interpolation_mode,
             padding_mode);
}