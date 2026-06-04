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

// On CUDA, data is already on device — Host2Device is a no-op passthrough
void py_cuda::cudaHost2DeviceOp(tpu::Host2DeviceOp op) {
  int num = module::getNumElements(op.getOutput());
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), getCudaData(op.getInput()),
                        num * sizeof(int), cudaMemcpyDeviceToDevice));
}
