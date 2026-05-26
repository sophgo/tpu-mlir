//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaMaxPoolingIndicesBwdOp(top::MaxPoolingIndicesBwdOp op) {
  auto out_shape = module::getShape(op.getGradOutput());
  int n = out_shape[0], c = out_shape[1], oh = out_shape[2], ow = out_shape[3];
  int num_elems = n * c * oh * ow;
  int in_num = module::getNumElements(op.getGradInput());
  auto in_shape = module::getShape(op.getGradInput());
  int ih = in_shape[2], iw = in_shape[3];
  auto ks = *module::getI64Array(op.getKernelShape());
  auto st = *module::getI64Array(op.getStrides());
  auto pd = *module::getI64Array(op.getPads());
  int kw = ks[1], sh = st[0], sw = st[1], ph = pd[0], pw = pd[1];

  std::vector<float> go(num_elems), idx(num_elems);
  if (getCudaType(op.getGradOutput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(go.data(), getCudaData(op.getGradOutput()),
                          num_elems * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getGradOutput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(go.data(), tmp.get(), num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  if (getCudaType(op.getIndices()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(idx.data(), getCudaData(op.getIndices()),
                          num_elems * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getIndices(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(idx.data(), tmp.get(), num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  std::vector<float> gi(in_num, 0.0f);
  for (int nn = 0; nn < n; nn++)
    for (int cc = 0; cc < c; cc++)
      for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
          int oi = ((nn * c + cc) * oh + oy) * ow + ox;
          int max_pos = (int)idx[oi];
          int pos_h = oy * sh - ph + max_pos / kw;
          int pos_w = ox * sw - pw + max_pos % kw;
          if (pos_h >= 0 && pos_h < ih && pos_w >= 0 && pos_w < iw) {
            int ii = ((nn * c + cc) * ih + pos_h) * iw + pos_w;
            gi[ii] = go[oi];
          }
        }

  auto out_f32 = cuda_malloc(in_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), gi.data(), in_num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getGradInput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getGradInput()), in_num,
                      cuda::DT_F32, getCudaType(op.getGradInput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getGradInput()), out_f32.get(),
                          in_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}

void py_cuda::cudaMaxPoolingIndicesBwdOp(tpu::MaxPoolingIndicesBwdOp op) {
  auto out_shape = module::getShape(op.getGradOutput());
  int n = out_shape[0], c = out_shape[1], oh = out_shape[2], ow = out_shape[3];
  int num_elems = n * c * oh * ow;
  int in_num = module::getNumElements(op.getGradInput());
  auto in_shape = module::getShape(op.getGradInput());
  int ih = in_shape[2], iw = in_shape[3];
  auto ks = *module::getI64Array(op.getKernelShape());
  auto st = *module::getI64Array(op.getStrides());
  auto pd = *module::getI64Array(op.getPads());
  int kw = ks[1], sh = st[0], sw = st[1], ph = pd[0], pw = pd[1];

  std::vector<float> go(num_elems), idx(num_elems);
  if (getCudaType(op.getGradOutput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(go.data(), getCudaData(op.getGradOutput()),
                          num_elems * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getGradOutput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(go.data(), tmp.get(), num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  if (getCudaType(op.getIndices()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(idx.data(), getCudaData(op.getIndices()),
                          num_elems * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getIndices(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(idx.data(), tmp.get(), num_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  std::vector<float> gi(in_num, 0.0f);
  for (int nn = 0; nn < n; nn++)
    for (int cc = 0; cc < c; cc++)
      for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
          int oi = ((nn * c + cc) * oh + oy) * ow + ox;
          int max_pos = (int)idx[oi];
          int pos_h = oy * sh - ph + max_pos / kw;
          int pos_w = ox * sw - pw + max_pos % kw;
          if (pos_h >= 0 && pos_h < ih && pos_w >= 0 && pos_w < iw) {
            int ii = ((nn * c + cc) * ih + pos_h) * iw + pos_w;
            gi[ii] = go[oi];
          }
        }

  auto out_f32 = cuda_malloc(in_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), gi.data(), in_num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getGradInput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getGradInput()), in_num,
                      cuda::DT_F32, getCudaType(op.getGradInput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getGradInput()), out_f32.get(),
                          in_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}
