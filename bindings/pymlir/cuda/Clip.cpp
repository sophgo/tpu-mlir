//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaClipOp(top::ClipOp op) {
  auto in = op.getInputs();
  auto out = op.getOutput();

  int64_t n, c, h, w;
  module::getNCHW(in, n, c, h, w, false);

  float clip_min = static_cast<float>(op.getMin().convertToDouble());
  float clip_max = static_cast<float>(op.getMax().convertToDouble());

  auto input = getCudaData(in);
  auto output = getCudaData(out);

  cuda::clip4DF32(input, output, clip_min, clip_max, n, c, h, w);
}

void py_cuda::cudaClipOp(tpu::ClipOp op) {
  auto in = op.getInput();
  auto out = op.getOutput();
  auto num_elements = module::getNumElements(out);
  auto in_type = module::getStorageType(in);
  auto out_type = module::getStorageType(out);
  int64_t n, c, h, w;
  module::getNCHW(in, n, c, h, w, false);

  double clip_min_d = op.getMin().convertToDouble();
  double clip_max_d = op.getMax().convertToDouble();
  float clip_min = static_cast<float>(clip_min_d);
  float clip_max = static_cast<float>(clip_max_d);

  // 如果输入输出都是 FP32，直接裁剪
  if (in_type.isF32() && out_type.isF32()) {
    auto input = getCudaData(in);
    auto output = getCudaData(out);
    cuda::clip4DF32(input, output, clip_min, clip_max, n, c, h, w);
    return;
  }

  // 否则，将输入转换为 FP32，裁剪后再转换回原类型
  auto input_f32 = newCudaData(in, cuda::DT_F32);
  auto output_f32 = cuda_malloc(num_elements * sizeof(float));
  cuda::clip4DF32(input_f32.get(), output_f32.get(), clip_min, clip_max, n, c, h, w);
  input_f32.reset();

  // 将 FP32 结果转换为输出类型
  cuda::convertType(output_f32.get(), getCudaData(out), num_elements,
                    cuda::DT_F32, getCudaType(out));
  output_f32.reset();
}