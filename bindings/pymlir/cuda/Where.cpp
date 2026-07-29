//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaWhereOp(top::WhereOp op) {
  int out_num = module::getNumElements(op.getOutput());
  auto c_shape = module::getShape(op.getCond());
  auto t_shape = module::getShape(op.getTbrn());
  auto f_shape = module::getShape(op.getFbrn());
  auto out_shape = module::getShape(op.getOutput());
  int rank = out_shape.size();

  // Read inputs as F32
  std::vector<float> cond(out_num), tbrn(out_num), fbrn(out_num);
  int c_num = module::getNumElements(op.getCond());
  int t_num = module::getNumElements(op.getTbrn());
  int f_num = module::getNumElements(op.getFbrn());
  if (getCudaType(op.getCond()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(cond.data(), getCudaData(op.getCond()),
                          c_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getCond(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(cond.data(), tmp.get(),
                          c_num * sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (getCudaType(op.getTbrn()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(tbrn.data(), getCudaData(op.getTbrn()),
                          t_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getTbrn(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(tbrn.data(), tmp.get(),
                          t_num * sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (getCudaType(op.getFbrn()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(fbrn.data(), getCudaData(op.getFbrn()),
                          f_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getFbrn(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(fbrn.data(), tmp.get(),
                          f_num * sizeof(float), cudaMemcpyDeviceToHost));
  }

  std::vector<float> output(out_num);
  for (int i = 0; i < out_num; i++) {
    int rem = i, ci = 0, ti = 0, fi = 0, cs = 1, ts = 1, fs = 1;
    for (int d = rank - 1; d >= 0; d--) {
      int coord = rem % out_shape[d]; rem /= out_shape[d];
      ci += (coord % c_shape[d]) * cs; cs *= c_shape[d];
      ti += (coord % t_shape[d]) * ts; ts *= t_shape[d];
      fi += (coord % f_shape[d]) * fs; fs *= f_shape[d];
    }
    output[i] = (cond[ci] != 0.0f) ? tbrn[ti] : fbrn[fi];
  }

  auto out_f32 = cuda_malloc(out_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), output.data(),
                        out_num * sizeof(float), cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), out_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                          out_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}
