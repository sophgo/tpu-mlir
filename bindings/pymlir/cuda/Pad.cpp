#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaPadOp(tpu::PadOp op) {
  auto pad_mode = op.getMode();
  auto pad_val = op.getVal().convertToDouble();
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto in_shape = module::getShape(op.getInput());
  auto pads = module::getI64Array(op.getPaddings());
  if (in_shape.size() != 4 || !module::isNone(op.getPaddingsT())) {
    llvm_unreachable("Only support 4D pad with static padding now");
  }
  if ((*pads)[0] != 0 || (*pads)[4] != 0 || (*pads)[1] != 0 || (*pads)[5] != 0) {
    llvm_unreachable("Only support pad on H and W now");
  }
  if (pad_mode == tpu::PaddingMode::constant) {
    if (module::isUniformQuantized(op.getOutput())) {
      cuda::pad4D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
          (*pads)[2], (*pads)[6], (*pads)[3], (*pads)[7], sizeof(int8_t), pad_val);
    } else if (module::getStorageType(op.getOutput()).isF32()) {
      cuda::pad4D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
          (*pads)[2], (*pads)[6], (*pads)[3], (*pads)[7], sizeof(float), pad_val);
    } else {
      if (module::getStorageType(op.getOutput()).isFloat8E4M3FN()) {
        llvm_unreachable("Not Implemented");
      }
      auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
      auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
      cuda::pad4D(input_f32.get(), output_f32.get(), in_shape[0], in_shape[1], in_shape[2], in_shape[3],
          (*pads)[2], (*pads)[6], (*pads)[3], (*pads)[7], sizeof(float), pad_val);
      cuda::convertType(output_f32.get(), output, module::getNumElements(op.getOutput()), cuda::DT_F32, getCudaType(op.getOutput()));
      input_f32.reset();
      output_f32.reset();
    }
  } else {
    bool is_edge = pad_mode == tpu::PaddingMode::edge; // edge or reflect
    int tbytes = module::getDtypeSize(op.getOutput());
    cuda::pad4D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
        (*pads)[2], (*pads)[6], (*pads)[3], (*pads)[7], tbytes, is_edge);
  }
}

void py_cuda::cudaPadOp(top::PadOp op) {
  auto pad_mode = op.getMode();
  auto pad_val = op.getVal().convertToDouble();
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  auto in_shape = module::getShape(op.getInput());
  auto pads = module::getI64Array(op.getPaddings());
  if (in_shape.size() != 4 || op.getPaddingsT()) {
    llvm_unreachable("Only support 4D pad with static padding now");
  }
  if ((*pads)[0] != 0 || (*pads)[4] != 0 || (*pads)[1] != 0 || (*pads)[5] != 0) {
    llvm_unreachable("Only support pad on H and W now");
  }
  if (pad_mode == "constant") {
    cuda::pad4D(input, output, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
          (*pads)[2], (*pads)[6], (*pads)[3], (*pads)[7], sizeof(float), pad_val);
  } else {
    llvm_unreachable("Pad (not constant) Not Implemented");
  }
}