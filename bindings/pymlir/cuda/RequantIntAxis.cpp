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

// void py_cuda::cudaRequantIntAxisOp(tpu::RequantIntAxisOp op) {
//   void *input = getCudaData(op.getInput());
//   void *quant = getCudaData(op.getQuant());
//   void *output = getCudaData(op.getOutput());
//   auto shape = std::vector<int64_t>(module::getShape(op.getQuant()));
//   if (shape.size() != 4) {
//     // 5 for 1d conv, not support, 4 for 2d deconv, shape is 1, oc, 1, 2 or 3
//     // assume zp 0
//     UNREACHABLE_OP("quant shape size not equal to 4", op);
//   }
//   while (shape.size() < 6) {
//     shape.push_back(1);
//   }
//   auto out_stype = module::getStorageType(op.getOutput());
//   auto sign = !out_stype.isUnsignedInteger(8);
//   int64_t n, c, h, w;
//   module::getNCHW(op.getInput(), n, c, h, w);
//   auto multipliers = cuda_malloc(shape[1] * sizeof(int32_t));
//   auto shifts = cuda_malloc(shape[1] * sizeof(int32_t));
//   cuda::slice6D(quant, multipliers.get(), shape[0], shape[1], shape[2],
//                 shape[3], shape[4], shape[5], 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, shape[0], shape[1], shape[2],
//                 1, shape[4], shape[5], sizeof(int32_t));
//   cuda::slice6D(quant, shifts.get(), shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], 0,
//                 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, shape[0], shape[1], shape[2], 1, shape[4], shape[5],
//                 sizeof(int32_t));
//   cuda::neg(shifts.get(), shifts.get(), shape[1], cuda::DT_INT32);
//   cuda::requantInt8Perchannel(input, output, multipliers.get(), shifts.get(), n,
//                               c, h, w, sign);
// }
void py_cuda::cudaRequantIntAxisOp(tpu::RequantIntAxisOp op) {
  void *input = getCudaData(op.getInput());
  void *quant = getCudaData(op.getQuant());
  void *output = getCudaData(op.getOutput());


  auto input_shape = module::getShape(op.getInput());
  auto q_shape_raw = module::getShape(op.getQuant());
  int dims = input_shape.size();


  if (dims != 4 && dims != 5) {
    UNREACHABLE_OP("RequantIntAxis only supports 4D or 5D tensors", op);
  }


  std::vector<int64_t> shape = q_shape_raw;
  while (shape.size() < 6) {
    shape.push_back(1);
  }

  auto out_stype = module::getStorageType(op.getOutput());
  auto sign = !out_stype.isUnsignedInteger(8);
  bool qdm = op.getQuantMode() == tpu::RequantMode::QDM;


  // 4D (1, oc, 1, 3) -> index 3; 5D (1, oc, 1, 1, 3) -> index 4
  int triplet_idx = (dims == 4) ? 3 : 4;
  int64_t oc = shape[1];

  auto multipliers = cuda_malloc(oc * sizeof(int32_t));
  auto shifts = cuda_malloc(oc * sizeof(int32_t));


  int64_t sz[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]};
  sz[triplet_idx] = 1;


  cuda::slice6D(quant, multipliers.get(), shape[0], shape[1], shape[2],
                shape[3], shape[4], shape[5], 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sizeof(int32_t));


  int64_t s_off[6] = {0, 0, 0, 0, 0, 0};
  s_off[triplet_idx] = 1;
  cuda::slice6D(quant, shifts.get(), shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                s_off[0], s_off[1], s_off[2], s_off[3], s_off[4], s_off[5],
                1, 1, 1, 1, 1, 1, sz[0], sz[1], sz[2], sz[3], sz[4], sz[5], sizeof(int32_t));


  cuda::neg(shifts.get(), shifts.get(), oc, cuda::DT_INT32);


  if (dims == 4) {
    int64_t n = input_shape[0], c = input_shape[1], h = input_shape[2], w = input_shape[3];
    cuda::requantInt8Perchannel(input, output, multipliers.get(), shifts.get(), n,
                                c, h, w, sign, qdm, false);
  } else {

    int64_t n = input_shape[0], c = input_shape[1], d = input_shape[2],
            h = input_shape[3], w = input_shape[4];

    cuda::requantInt8Perchannel_3d(input, output, multipliers.get(), shifts.get(),
                                   n, c, h, w, d, sign, qdm, false);
  }
}
