//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaTriluOp(top::TriluOp op) {
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  int diagonal = op.getDiagonal();
  bool upper = op.getUpper();
  int batch = (int)n * (int)c;

  auto stype = module::getStorageType(op.getInput());
  if (stype.isF32()) {
    cuda::trilu(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                batch, (int)h, (int)w, (int)w, diagonal, upper);
  } else {
    int num_elem = n * c * h * w;
    auto out_f32 = cuda_malloc(num_elem * sizeof(float));
    auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    cuda::trilu(in_f32.get(), out_f32.get(), batch, (int)h, (int)w,
                (int)w, diagonal, upper);
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), num_elem,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
