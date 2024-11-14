//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <queue>
#include <vector>

LogicalResult tpu::ShapeArithOp::init(InferenceParameter &p) {
  auto binary = new Binary();
  auto lhs_shape = module::getShape(getInputs()[0]);
  auto rhs_shape = module::getShape(getInputs()[1]);
  algorithm alg;
  std::map<std::string, algorithm> map_mode = {
      {"Add", algorithm::binary_add},
      {"Sub", algorithm::binary_sub},
      {"Mul", algorithm::binary_mul},
      {"Div", algorithm::binary_div},
      {"Less", algorithm::binary_lt},
      {"Greater", algorithm::binary_gt},
      {"LessOrEqual", algorithm::binary_le},
      {"GreaterOrEqual", algorithm::binary_ge},
      {"Min", algorithm::binary_min},
      {"Max", algorithm::binary_max},
      {"Equal", algorithm::binary_eq},
      {"NotEqual", algorithm::binary_ne}};
  auto iter = map_mode.find(getType().str());
  if (iter != map_mode.end()) {
    alg = iter->second;
  } else {
    llvm_unreachable("Get unsupported arith type!");
  }

  (*binary)
      .hs(p.inputs[0], p.inputs[1], lhs_shape, rhs_shape)
      .dst(p.outputs[0], module::getShape(getOutput()))
      .algorithem(alg)
      .setup();
  p.handle = (void *)binary;
  return success();
}
void tpu::ShapeArithOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::ShapeArithOp::inference(InferenceParameter &p) {
  auto binary = (Binary *)p.handle;
  binary->run();

  std::string op_type = getType().str();
  if (op_type == "Div") {
    auto ops = getOperands();
    if (backend::BM168x::getDataType(ops[0]) == DTYPE_INT32 &&
        backend::BM168x::getDataType(ops[1]) == DTYPE_INT32) {
      auto num_elem = module::getNumElements(getOutput());

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = std::floor(p.outputs[0][i]);
      }
    }
  }

  return success();
}

bool tpu::ShapeArithOp::support_multi_core() { return false; }
