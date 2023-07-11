//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PermuteOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto attr = parseParam();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(input_spec->at(0), attr.in_shape_fix);
  BM168x::fix_shape(output_spec->at(0), attr.out_shape_fix);
  transpose_param_t param = {0};
  param.spec.buffer_global_addr = module::getAddress(getBuffer());
  for (int i = 0; i < attr.order_fix.size(); i++) {
    param.spec.order[i] = attr.order_fix[i];
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
  if (!buffer)
    return sizeof(transpose_spec_t);
  transpose_spec_t spec;
  auto perm = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  int dims = in_shape.size();
  spec.buffer_global_addr = module::getAddress(getBuffer());
  for (int i = 0; i < dims; i++) {
    spec.order[i] = perm->at(i);
  }

  int input_neuron_tensors = 0;
  auto op = getOperation();
  for (auto v : op->getOperands()) {
    if (!isa<top::WeightOp, top::NoneOp, tpu::BufferOp>(v.getDefiningOp())) {
      input_neuron_tensors++;
    }
  }
  spec.is_dynamic = input_neuron_tensors > 1;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::PermuteOp::get_fw_type_bm1684x() {
  return FW_BMNET_TRANSPOSE;
}
