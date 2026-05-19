// //===----------------------------------------------------------------------===//
// //
// // Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// //
// // TPU-MLIR is licensed under the 2-Clause BSD License except for the
// // third-party components.
// //
// //===----------------------------------------------------------------------===//

// #include "../pycuda.h"
// #include "cuda_helper.h"

// // ======================================
// // top::AddConstOp (浮点，不涉及量化)
// // ======================================
// void py_cuda::cudaAddConstOp(top::AddConstOp op) {
//   auto in = op.getInput();
//   auto out = op.getOutput();
//   int64_t n, c, h, w;
//   module::getNCHW(in, n, c, h, w, false);
//   float const_val = static_cast<float>(op.getConstVal().convertToDouble());
//   bool do_relu = op.getDoRelu();

//   auto input = getCudaData(in);
//   auto output = getCudaData(out);
//   cuda::addConst4DF32(input, output, const_val, do_relu, n, c, h, w);
// }

// // ======================================
// // tpu::AddConstOp (暂不支持，留空)
// // ======================================
// void py_cuda::cudaAddConstOp(tpu::AddConstOp op) {
//   UNREACHABLE_OP("tpu::AddConstOp not implemented yet", op);
// }
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

void py_cuda::cudaAddConstOp(top::AddConstOp op) {
  auto in = op.getInput();
  auto out = op.getOutput();
  int64_t n, c, h, w;
  module::getNCHW(in, n, c, h, w, false);
  float const_val = static_cast<float>(op.getConstVal().convertToDouble());
  bool do_relu = op.getDoRelu();

  auto input = getCudaData(in);
  auto output = getCudaData(out);
  cuda::addConst4DF32(input, output, const_val, do_relu, n, c, h, w);
}

void py_cuda::cudaAddConstOp(tpu::AddConstOp op) {
  auto in = op.getInput();
  auto out = op.getOutput();
  auto num_elements = module::getNumElements(out);
  auto out_stype = module::getStorageType(out);
  int64_t n, c, h, w;
  module::getNCHW(in, n, c, h, w, false);
  float const_val = static_cast<float>(op.getConstVal().convertToDouble());
  bool do_relu = op.getDoRelu();

  // 处理均匀量化 (INT8/INT16)
  if (module::isUniformQuantized(out)) {
    // 将输入转换为 FP32
    auto input_f32 = newCudaData(in, cuda::DT_F32);
    // 构造常量张量（全为 const_val）
    auto const_data = std::make_shared<std::vector<float>>(num_elements, const_val);
    auto const_tensor = cuda_malloc(num_elements * sizeof(float));
    CHECK_CUDA(cudaMemcpy(const_tensor.get(), const_data->data(),
                          num_elements * sizeof(float), cudaMemcpyHostToDevice));
    // 执行 FP32 加法
    auto output_f32 = cuda_malloc(num_elements * sizeof(float));
    cuda::add4DF32(input_f32.get(), 1.0, const_tensor.get(), 1.0, output_f32.get(),
                   false, n, c, h, w, n, c, h, w, n, c, h, w);
    // 获取量化参数（标量）
    int32_t multiplier = static_cast<int32_t>(op.getMultiplier());
    int32_t rshift = static_cast<int32_t>(op.getRshift());
    bool sign = !out_stype.isUnsignedInteger(8);
    bool relu = sign && do_relu;
    if (out_stype.isInteger(16)) {
      auto out_i32 = cuda_malloc(num_elements * sizeof(int32_t));
      cuda::convertType(output_f32.get(), out_i32.get(), num_elements,
                        cuda::DT_F32, cuda::DT_INT32, cuda::RD_HALF_UP);
      cuda::requantInt16(out_i32.get(), getCudaData(out), multiplier,
                         rshift, num_elements, relu);
    } else {
      cuda::requantInt8(output_f32.get(), getCudaData(out), multiplier,
                        rshift, num_elements, sign, false, relu);
    }
    input_f32.reset();
    const_tensor.reset();
    output_f32.reset();
    return;
  }

  // 处理 FP8 输出
  if (module::getStorageType(out).isFloat8E4M3FN()) {
    auto input_f32 = newCudaData(in, cuda::DT_F32);
    auto output_f32 = cuda_malloc(num_elements * sizeof(float));
    cuda::addConst4DF32(input_f32.get(), output_f32.get(), const_val, do_relu,
                        n, c, h, w);
    double scale = op.getF8Scale().convertToDouble();
    cuda::requantF8(output_f32.get(), getCudaData(out), scale,
                    n, c, h, w, do_relu);
    input_f32.reset();
    output_f32.reset();
    return;
  }

  // 浮点路径
  auto input = getCudaData(in);
  auto output = getCudaData(out);
  if (module::getStorageType(in).isF32()) {
    cuda::addConst4DF32(input, output, const_val, do_relu, n, c, h, w);
  } else {
    auto input_f32 = newCudaData(in, cuda::DT_F32);
    auto output_f32 = cuda_malloc(num_elements * sizeof(float));
    cuda::addConst4DF32(input_f32.get(), output_f32.get(), const_val, do_relu,
                        n, c, h, w);
    cuda::convertType(output_f32.get(), output, num_elements,
                      cuda::DT_F32, getCudaType(out));
    input_f32.reset();
    output_f32.reset();
  }
}