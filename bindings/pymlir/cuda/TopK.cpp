//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <vector>

void py_cuda::cudaTopKOp(top::TopKOp op) {
  auto in_shape = module::getShape(op.getInput());
  int in_num = module::getNumElements(op.getInput());
  int axis = op.getAxis();
  int dims = in_shape.size();
  if (axis < 0) axis += dims;
  int K = op.getK();
  bool largest = op.getLargest();

  std::vector<float> data(in_num);
  CHECK_CUDA(cudaMemcpy(data.data(), getCudaData(op.getInput()),
                        in_num * sizeof(float), cudaMemcpyDeviceToHost));

  int outer = 1;
  for (int i = 0; i < axis; i++) outer *= in_shape[i];
  int axis_len = in_shape[axis];

  int out_num = module::getNumElements(op.getValues());
  std::vector<float> val_out(out_num);
  std::vector<float> idx_out(out_num);

  for (int o = 0; o < outer; o++) {
    std::vector<std::pair<int, float>> result;
    topk_indices(result, data.data() + o * axis_len, axis_len, K, largest);
    for (int k = 0; k < K; k++) {
      val_out[o * K + k] = result[k].second;
      idx_out[o * K + k] = (float)result[k].first;
    }
  }

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getValues()), val_out.data(),
                        out_num * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getIndices()), idx_out.data(),
                        out_num * sizeof(float), cudaMemcpyHostToDevice));
}

void py_cuda::cudaTopKOp(tpu::TopKOp op) {
  auto in_shape = module::getShape(op.getInput());
  int in_num = module::getNumElements(op.getInput());
  int axis = op.getAxis();
  int dims = in_shape.size();
  if (axis < 0) axis += dims;
  int K = op.getK();
  bool largest = op.getLargest();

  auto stype = module::getStorageType(op.getInput());
  std::vector<float> data(in_num);
  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(data.data(), getCudaData(op.getInput()),
                          in_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto in_f32 = cuda_malloc(in_num * sizeof(float));
    cuda::convertType(getCudaData(op.getInput()), in_f32.get(), in_num,
                      getCudaType(op.getInput()), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(data.data(), in_f32.get(),
                          in_num * sizeof(float), cudaMemcpyDeviceToHost));
  }

  int outer = 1;
  for (int i = 0; i < axis; i++) outer *= in_shape[i];
  int axis_len = in_shape[axis];

  int out_num = module::getNumElements(op.getValues());
  std::vector<float> val_out(out_num);
  std::vector<float> idx_out(out_num);

  for (int o = 0; o < outer; o++) {
    std::vector<std::pair<int, float>> result;
    topk_indices(result, data.data() + o * axis_len, axis_len, K, largest);
    for (int k = 0; k < K; k++) {
      val_out[o * K + k] = result[k].second;
      idx_out[o * K + k] = (float)result[k].first;
    }
  }

  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getValues()), val_out.data(),
                          out_num * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getIndices()), idx_out.data(),
                          out_num * sizeof(float), cudaMemcpyHostToDevice));
  } else {
    auto val_f32 = cuda_malloc(out_num * sizeof(float));
    auto idx_f32 = cuda_malloc(out_num * sizeof(float));
    CHECK_CUDA(cudaMemcpy(val_f32.get(), val_out.data(),
                          out_num * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(idx_f32.get(), idx_out.data(),
                          out_num * sizeof(float), cudaMemcpyHostToDevice));
    cuda::convertType(val_f32.get(), getCudaData(op.getValues()), out_num,
                      cuda::DT_F32, getCudaType(op.getValues()));
    cuda::convertType(idx_f32.get(), getCudaData(op.getIndices()), out_num,
                      cuda::DT_F32, getCudaType(op.getIndices()));
  }
}
