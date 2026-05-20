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

void py_cuda::cudaDevice2HostOp(tpu::Device2HostOp op) {
  auto num = module::getNumElements(op.getOutput());
  auto bytes = num * module::getDtypeSize(op.getOutput());
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()),
                        getCudaData(op.getInput()), bytes,
                        cudaMemcpyDeviceToDevice));
}
