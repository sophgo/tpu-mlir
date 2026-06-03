//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaScaleLutOp(top::ScaleLutOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);

  auto sc_attr = module::getF64Array(op.getScale());
  auto bi_attr = module::getF64Array(op.getBias());
  std::vector<float> sc(c), bi(c);
  for (int i = 0; i < c; i++) {
    sc[i] = sc_attr->at(i % sc_attr->size());
    bi[i] = bi_attr->at(i % bi_attr->size());
  }
  auto sc_d = cuda_malloc(c * sizeof(float));
  auto bi_d = cuda_malloc(c * sizeof(float));
  CHECK_CUDA(cudaMemcpy(sc_d.get(), sc.data(), c * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(bi_d.get(), bi.data(), c * sizeof(float), cudaMemcpyHostToDevice));
  cuda::scaleLut(input, output, sc_d.get(), bi_d.get(), n, c, h * w);
}

void py_cuda::cudaScaleLutOp(tpu::ScaleLutOp op) {
  int64_t n, c, h, w;
  module::getNCHW(op.getInput(), n, c, h, w);
  auto num_elements = n * c * h * w;

  auto sc_attr = module::getF64Array(op.getScale());
  auto bi_attr = module::getF64Array(op.getBias());
  std::vector<float> sc(c), bi(c);
  for (int i = 0; i < c; i++) {
    sc[i] = sc_attr->at(i % sc_attr->size());
    bi[i] = bi_attr->at(i % bi_attr->size());
  }
  auto sc_d = cuda_malloc(c * sizeof(float));
  auto bi_d = cuda_malloc(c * sizeof(float));
  CHECK_CUDA(cudaMemcpy(sc_d.get(), sc.data(), c * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(bi_d.get(), bi.data(), c * sizeof(float), cudaMemcpyHostToDevice));

  auto stype = module::getStorageType(op.getInput());
  if (stype.isF32()) {
    cuda::scaleLut(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                   sc_d.get(), bi_d.get(), n, c, h * w);
  } else {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    auto output_f32 = cuda_malloc(num_elements * sizeof(float));
    cuda::scaleLut(input_f32.get(), output_f32.get(), sc_d.get(), bi_d.get(), n, c, h * w);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), num_elements,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
