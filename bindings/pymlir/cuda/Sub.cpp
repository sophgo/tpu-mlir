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

void py_cuda::cudaSubOp(tpu::SubOp op) {
  auto asym = module::isAsymmetric();
  auto out = op.getOutput();
  if (asym || op.getInputs().size() != 2)
    UNREACHABLE_OP("Not Implemented", op);
  auto shape0 = module::getShape(op.getInputs()[0]);
  auto shape1 = module::getShape(op.getInputs()[1]);
  if (shape0.size() != shape1.size()) {
    UNREACHABLE_OP("Not supported", op);
  }
  if (module::isUniformQuantized(op.getInputs()[0])) {
    auto in0 = op.getInputs()[0];
    auto in1 = op.getInputs()[1];
    auto in0_unsign = module::getStorageType(in0).isUnsignedInteger();
    auto in1_unsign = module::getStorageType(in1).isUnsignedInteger();
    auto out_unsign = module::getStorageType(out).isUnsignedInteger();
    if (out_unsign)
      UNREACHABLE_OP("Not supported, and not possible", op);
    int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
    module::getNCHW(in0, n0, c0, h0, w0, false);
    module::getNCHW(in1, n1, c1, h1, w1, false);
    module::getNCHW(out, n2, c2, h2, w2, false);
    auto multiplier_v = module::getI64Array(op.getMultipliers(), 2, 1);
    auto rshift_v = module::getI64Array(op.getRshifts(), 2, 0);

    cuda::sub4DInt8(getCudaData(in0), in0_unsign, multiplier_v->at(0), rshift_v->at(0), getCudaData(in1), in1_unsign, multiplier_v->at(1), rshift_v->at(1), getCudaData(out), out_unsign, op.getDoRelu(), op.getIsReverse(),
                    n0, c0, h0, w0,
                    n1, c1, h1, w1,
                    n2, c2, h2, w2);
  } else if (module::getStorageType(op.getInputs()[0]).isF32()) {
    auto in0 = op.getInputs()[0];
    auto in1 = op.getInputs()[1];
    auto input0 = getCudaData(in0);
    auto input1 = getCudaData(in1);
    auto output = getCudaData(out);
    int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
    module::getNCHW(in0, n0, c0, h0, w0, false);
    module::getNCHW(in1, n1, c1, h1, w1, false);
    module::getNCHW(out, n2, c2, h2, w2, false);
    cuda::sub4DF32(input0, input1, output, op.getDoRelu(), op.getIsReverse(),
                    n0, c0, h0, w0,
                    n1, c1, h1, w1,
                    n2, c2, h2, w2);
  } else {
    auto in0 = op.getInputs()[0];
    auto in1 = op.getInputs()[1];
    auto input0_f32 = newCudaData(in0, cuda::DT_F32);
    auto input1_f32 = newCudaData(in1, cuda::DT_F32);
    auto output_f32 = newCudaData(out, cuda::DT_F32);
    int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
    module::getNCHW(in0, n0, c0, h0, w0, false);
    module::getNCHW(in1, n1, c1, h1, w1, false);
    module::getNCHW(out, n2, c2, h2, w2, false);
    cuda::sub4DF32(input0_f32.get(), input1_f32.get(), output_f32.get(), op.getDoRelu(), op.getIsReverse(),
                    n0, c0, h0, w0,
                    n1, c1, h1, w1,
                    n2, c2, h2, w2);
    cuda::convertType(output_f32.get(), getCudaData(out), module::getNumElements(out), cuda::DT_F32,
                getCudaType(op.getOutput()));
    input0_f32.reset();
    input1_f32.reset();
    output_f32.reset();
  }
}

void py_cuda::cudaSubOp(top::SubOp op) {
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
  cuda::sub4DF32(input0, input1, output, op.getDoRelu(), op.getIsReverse(),
                  n0, c0, h0, w0,
                  n1, c1, h1, w1,
                  n2, c2, h2, w2);
}
