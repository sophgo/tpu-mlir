//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;


llvm::ArrayRef<int64_t> shape_expand_dim(llvm::ArrayRef<int64_t> shape, int dims) {
  int diff = dims - shape.size();
  if (diff == 0)
    return shape;
  std::vector<int64_t> shape_v(shape.begin(), shape.end());
  shape_v.insert(shape_v.begin(), diff, 1);
  auto shape_ed = llvm::ArrayRef<int64_t>(shape_v);
  return shape_ed;
}

LogicalResult tpu::MulOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  auto in0_shape = Module::getShape(inputs()[0]);
  auto in1_shape = Module::getShape(inputs()[1]);
  int dims = std::max(in0_shape.size(), in1_shape.size());
  auto input0_shape = shape_expand_dim(in0_shape, dims);
  auto input1_shape = shape_expand_dim(in1_shape, dims);
  (*binary)
      .lhs(p.inputs[0], input0_shape)
      .rhs(p.inputs[1], input1_shape)
      .dst(p.outputs[0], Module::getShape(output()))
      .do_relu(do_relu())
      .relu_limit(relu_limit().convertToDouble())
      .algorithem(algorithm::binary_mul)
      .setup();
  p.handle = (void *)binary;
  return success();
}

void tpu::MulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::MulOp::inference(InferenceParameter &p) {
  auto module = Module::getModuleOp(getOperation());
  int nInputs = inputs().size();
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  auto asym = Module::getAsymmetric(module);
  auto binary = (Binary *)p.handle;
  binary->run();
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (out_type.isInteger(32)) {
    return success();
  } else if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      double sum = p.outputs[0][i];
      sum = applyMultiplierAndRShift(sum, multiplier(), rshift());
      p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(sum)
                                                      : Quant::to_int8(sum);
    }
  } else {
    llvm_unreachable("MulOp asymmetric use FP32");
  }
  return success();
}

LogicalResult tpu::MulOp::LocalGenSupport() {
  // BackwardH and BackwardN can not handle more than one input right now.
  // The same n_slice and h_slice value will propagate to each inputs.
  // Thus, the local layer is only safe when we do not need to slice n and h
  // dimensions.
  auto out_shape = Module::getShape(output());
  auto lhs_shape = Module::getShape(inputs()[0]);
  auto rhs_shape = Module::getShape(inputs()[1]);
  if (getOperand(1).getDefiningOp() &&
      isa<top::WeightOp>(getOperand(1).getDefiningOp()))
    return failure();
  // left align
  switch (out_shape.size()) {
  case 2:
    if (lhs_shape[0] != rhs_shape[0])
      return failure();
  case 3:
  case 4:
    if (lhs_shape[0] != rhs_shape[0])
      return failure();
    if (lhs_shape[2] != rhs_shape[2])
      return failure();
  default:
    success();
  }
  return success();
}
