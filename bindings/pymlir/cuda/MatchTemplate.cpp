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

void py_cuda::cudaMatchTemplateOp(top::MatchTemplateOp op) {
  auto input = getCudaData(op.getInput());
  auto templ = getCudaData(op.getMatch());
  auto output = getCudaData(op.getOutput());

  auto i_shape = module::getShape(op.getInput());
  auto t_shape = module::getShape(op.getMatch());
  auto o_shape = module::getShape(op.getOutput());

  int iH = i_shape[0], iW = i_shape[1];
  int tH = t_shape[0], tW = t_shape[1];
  int oH = o_shape[0], oW = o_shape[1];
  int mode = (op.getMode().str() == "TM_CCOEFF_NORMED") ? 1 : 0;

  cuda::matchTemplate(input, templ, output, iH, iW, tH, tW, oH, oW, mode);
}

void py_cuda::cudaMatchTemplateOp(tpu::MatchTemplateOp op) {
  auto input = getCudaData(op.getInput());
  auto templ = getCudaData(op.getMatch());
  auto output = getCudaData(op.getOutput());

  auto i_shape = module::getShape(op.getInput());
  auto t_shape = module::getShape(op.getMatch());
  auto o_shape = module::getShape(op.getOutput());

  int iH = i_shape[0], iW = i_shape[1];
  int tH = t_shape[0], tW = t_shape[1];
  int oH = o_shape[0], oW = o_shape[1];
  int mode = (op.getMode().str() == "TM_CCOEFF_NORMED") ? 1 : 0;

  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    cuda::matchTemplate(input, templ, output, iH, iW, tH, tW, oH, oW, mode);
  } else {
    auto num = module::getNumElements(op.getOutput());
    auto output_f32 = cuda_malloc(num * sizeof(float));
    cuda::matchTemplate(input, templ, output_f32.get(), iH, iW, tH, tW, oH, oW, mode);
    cuda::convertType(output_f32.get(), output, num, cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
