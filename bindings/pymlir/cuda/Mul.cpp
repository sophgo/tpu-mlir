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

void py_cuda::cudaMulOp(tpu::MulOp op) {
  if (op.getInputs().size() != 2)
    UNREACHABLE_OP("Not Implemented", op);
  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(op.getInputs()[0], n0, c0, h0, w0, false);
  module::getNCHW(op.getInputs()[1], n1, c1, h1, w1, false);
  module::getNCHW(op.getOutput(), n2, c2, h2, w2, false);
  auto shape0 = module::getShape(op.getInputs()[0]);
  auto shape1 = module::getShape(op.getInputs()[1]);
  if (shape0.size() != shape1.size()) {
    UNREACHABLE_OP("Not supported", op);
  }
  if (module::isUniformQuantized(op.getInputs()[0])) {
    auto multiplier = op.getMultiplier();
    auto rshift = op.getRshift();

    cuda::mulInt8(getCudaData(op.getInputs()[0]), getCudaData(op.getInputs()[1]), getCudaData(op.getOutput()), n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2,
            !module::getStorageType(op.getInputs()[0]).isUnsignedInteger(), !module::getStorageType(op.getInputs()[1]).isUnsignedInteger(), !module::getStorageType(op.getOutput()).isUnsignedInteger(), multiplier, rshift,
             false, op.getDoRelu());
  } else if (module::getStorageType(op.getInputs()[0]).isF32()) {
    cuda::mul4DF32(getCudaData(op.getInputs()[0]), getCudaData(op.getInputs()[1]), getCudaData(op.getOutput()), op.getDoRelu(),
                  n0, c0, h0, w0,
                  n1, c1, h1, w1,
                  n2, c2, h2, w2);
  } else {
    auto input0_f32 = newCudaData(op.getInputs()[0], cuda::DT_F32);
    auto input1_f32 = newCudaData(op.getInputs()[1], cuda::DT_F32);
    auto output_f32 = newCudaData(op.getOutput(), cuda::DT_F32);
    cuda::mul4DF32(input0_f32.get(), input1_f32.get(), output_f32.get(), op.getDoRelu(),
                  n0, c0, h0, w0,
                  n1, c1, h1, w1,
                  n2, c2, h2, w2);
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), module::getNumElements(op.getOutput()), cuda::DT_F32,
                      getCudaType(op.getOutput()));
    input0_f32.reset();
    input1_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaMulOp(top::MulOp op) {
  auto out = op.getOutput();
  auto num_inputs = op.getInputs().size();
  if (2 != num_inputs) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto in0 = op.getInputs()[0];
  auto in1 = op.getInputs()[1];
  auto input0 = getCudaData(in0);
  auto input1 = getCudaData(in1);
  auto output = getCudaData(out);
  auto shape0 = module::getShape(in0);
  auto shape1 = module::getShape(in1);
  if (shape0.size() != shape1.size()) {
    UNREACHABLE_OP("Not supported", op);
  }
  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(in0, n0, c0, h0, w0, false);
  module::getNCHW(in1, n1, c1, h1, w1, false);
  module::getNCHW(out, n2, c2, h2, w2, false);
  if (shape0.size() > 4 && (n0 != n1)) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  cuda::mul4DF32(input0, input1, output, op.getDoRelu(),
                  n0, c0, h0, w0,
                  n1, c1, h1, w1,
                  n2, c2, h2, w2);
}
