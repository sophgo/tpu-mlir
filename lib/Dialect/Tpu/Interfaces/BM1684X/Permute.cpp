//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"



using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PermuteOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  transpose_param_t param = {0};
  param.if_getting_buffer_size = 0;

  auto perm = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  int dims = in_shape.size();
  param.spec.buffer_global_addr = module::getAddress(getBuffer());
  for (int i = 0; i < dims; i++) {
    param.spec.order[i] = perm->at(i);
  }
  param.buffer_size_ptr = 0;
  BM168x::call_global_func("backend_api_transpose_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::PermuteOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(transpose_spec_t);
  transpose_spec_t spec;
  auto perm = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  int dims = in_shape.size();
  spec.buffer_global_addr = module::getAddress(getBuffer());
  for (int i = 0; i < dims; i++) {
    spec.order[i] =perm->at(i);
  }

  int input_neuron_tensors = 0;
  auto op = getOperation();
  for (auto v: op->getOperands()) {
    if (!isa<top::WeightOp, top::NoneOp>(v.getDefiningOp())) {
      input_neuron_tensors++;
    }
  }
  spec.is_dynamic = input_neuron_tensors > 1;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}
