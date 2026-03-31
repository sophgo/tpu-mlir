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

void py_cuda::cudaAddOp(tpu::AddOp op) {
  auto out = op.getOutput();
  auto num_inputs = op.getInputs().size();
  if (2 != num_inputs) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto in0 = op.getInputs()[0];
  auto in1 = op.getInputs()[1];
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
  if (module::isUniformQuantized(op.getOutput())) {
    auto input0 = getCudaData(in0);
    auto input1 = getCudaData(in1);
    auto output = getCudaData(out);
    if (module::isCV18xx()) {
      auto multiplier_v =
          module::getI64Array(op.getMultipliers(), op.getInputs().size(), 1);
      auto rshift_v = module::getI64Array(op.getRshifts(), 1, 0);
      cuda::cvAdd4DInt8(input0, input1, output, multiplier_v->at(0),
                        multiplier_v->at(1), rshift_v->at(0), op.getDoRelu(), n0,
                        c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2);
    } else {
      auto sign0 = !module::getStorageType(in0).isUnsignedInteger(8);
      auto sign1 = !module::getStorageType(in1).isUnsignedInteger(8);
      auto out_sign = !module::getStorageType(out).isUnsignedInteger(8);
      auto multiplier_v =
          module::getI64Array(op.getMultipliers(), op.getInputs().size(), 1);
      auto rshift_v =
          module::getI64Array(op.getRshifts(), op.getInputs().size(), 0);
      cuda::add4DInt8(input0, input1, output, multiplier_v->at(0),
                      multiplier_v->at(1), rshift_v->at(0), rshift_v->at(1),
                      sign0, sign1, out_sign, op.getDoRelu(), n0, c0, h0, w0, n1,
                      c1, h1, w1, n2, c2, h2, w2);
    }
  } else {
    if (module::getStorageType(out).isFloat8E4M3FN()) {
      auto in_scales = module::getF64Array(op.getF8Scales(), 2, 1.);
      auto input0 = newCudaData(in0, cuda::DT_F32);
      auto input1 = newCudaData(in1, cuda::DT_F32);
      auto input0_ui16 = cuda_malloc(module::getNumElements(in0) * sizeof(uint16_t));
      auto input1_ui16 = cuda_malloc(module::getNumElements(in1) * sizeof(uint16_t));
      auto output_f32 = cuda_malloc(module::getNumElements(out) * sizeof(float));
      auto output_ui16 = cuda_malloc(module::getNumElements(out) * sizeof(uint16_t));

      cuda::mulConst4DF32(input0.get(), F16(in_scales->at(0)), input0.get(), false,
                    n0, c0, h0, w0);
      cuda::mulConst4DF32(input1.get(), F16(in_scales->at(1)), input1.get(), false,
                    n1, c1, h1, w1);
      cuda::convertType(input0.get(), input0_ui16.get(), module::getNumElements(in0),
          cuda::DT_F32, cuda::DT_F16);
      cuda::convertType(input1.get(), input1_ui16.get(), module::getNumElements(in1),
          cuda::DT_F32, cuda::DT_F16);
      cuda::convertType(input0_ui16.get(), input0.get(), module::getNumElements(in0),
          cuda::DT_F16, cuda::DT_F32);
      cuda::convertType(input1_ui16.get(), input1.get(), module::getNumElements(in1),
          cuda::DT_F16, cuda::DT_F32);
      cuda::add4DF32(input0.get(), 1, input1.get(), 1, output_f32.get(), op.getDoRelu(),
                     n0, c0, h0, w0,
                     n1, c1, h1, w1,
                     n2, c2, h2, w2);
      cuda::convertType(output_f32.get(), output_ui16.get(), module::getNumElements(out),
          cuda::DT_F32, cuda::DT_F16);
      cuda::convertType(output_ui16.get(), output_f32.get(), module::getNumElements(out),
          cuda::DT_F16, cuda::DT_F32);
      cuda::requantF8(output_f32.get(), getCudaData(out), 1.0 , n2, c2, h2, w2, op.getDoRelu());
      input0.reset();
      input1.reset();
      output_f32.reset();
      output_ui16.reset();
      input0_ui16.reset();
      input1_ui16.reset();
    }
    else if (module::getStorageType(in0).isF32()) {
      auto input0 = getCudaData(in0);
      auto input1 = getCudaData(in1);
      auto output = getCudaData(out);
      cuda::add4DF32(input0, 1.0, input1, 1.0, output, op.getDoRelu(),
                     n0, c0, h0, w0,
                     n1, c1, h1, w1,
                     n2, c2, h2, w2);
      return;
    } else {
      auto input0 = newCudaData(in0, cuda::DT_F32);
      auto input1 = newCudaData(in1, cuda::DT_F32);
      auto output = cuda_malloc(module::getNumElements(out) * sizeof(float));
      cuda::add4DF32(input0.get(), 1.0, input1.get(), 1.0, output.get(), op.getDoRelu(),
                     n0, c0, h0, w0,
                     n1, c1, h1, w1,
                     n2, c2, h2, w2);
      cuda::convertType(output.get(), getCudaData(out), module::getNumElements(out), cuda::DT_F32,
                        getCudaType(out));
      input0.reset();
      input1.reset();
      output.reset();
      return;
    }
  }
}

void py_cuda::cudaAddOp(top::AddOp op) {
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
  cuda::add4DF32(input0, 1.0, input1, 1.0, output, op.getDoRelu(),
                  n0, c0, h0, w0,
                  n1, c1, h1, w1,
                  n2, c2, h2, w2);
}
