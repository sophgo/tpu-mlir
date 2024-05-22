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

// ======================================
// GlobalGenInterface
// ======================================
void tpu::ShapeArithOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeArithOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (buffer) {
    std::string op_type = getType().str();

    // TODO: Min Max GT LT GE LE SQRT ...
    if (op_type == "Add")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_ADD);
    else if (op_type == "Sub")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_SUB);
    else if (op_type == "Mul")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_MUL);
    else if (op_type == "Div")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_DIV);
    else if (op_type == "Less")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_LT);
    else if (op_type == "Greater")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_GT);
    else if (op_type == "LessOrEqual")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_LE);
    else if (op_type == "GreaterOrEqual")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_GE);
    else if (op_type == "Min")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_MIN);
    else if (op_type == "Max")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_MAX);
    else if (op_type == "Equal")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_EQ);
    else if (op_type == "NotEqual")
      return BM168x::dynamic_spec_to_buffer(buffer, BINARY_NE);
    else
      llvm_unreachable("Not supported now");
  }
  return sizeof(int);
}

int64_t tpu::ShapeArithOp::get_fw_type_bm1684x() { return FW_BMNET_SHAPE_OP; }
