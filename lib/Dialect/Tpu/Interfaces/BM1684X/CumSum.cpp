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

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  int dim;
  int N;
  int C;
  int H;
  int W;
  int dtype;
} cumsum_global_param_t;

typedef struct {
  int dim;
  int N;
  int C;
  int H;
  int W;
  int dtype;
} dyn_cumsum_global_param_t;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::CumSumOp::codegen_global_bm1684x() {
  cumsum_global_param_t p = {0};
  auto in_shape = module::getShape(getInput());
  p.input_addr = module::getAddress(getInput());
  p.output_addr = module::getAddress(getOutput());

  int64_t shape_size = in_shape.size();
  p.dim = getAxis();
  p.N = in_shape[0];
  p.C = 1;
  p.H = 1;
  p.W = 1;

  switch (shape_size) {
  case 2:
    p.C = in_shape[1];
    break;
  case 3:
    p.C = in_shape[1];
    p.H = in_shape[2];
    break;
  case 4:
    p.C = in_shape[1];
    p.H = in_shape[2];
    p.W = in_shape[3];
  default:
    break;
  }
  p.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_cumsum", &p,
                           sizeof(cumsum_global_param_t));
}

// =========================================
// Dyn GlobalGenInterface
// =========================================
int64_t tpu::CumSumOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_cumsum_global_param_t);
  dyn_cumsum_global_param_t p = {0};
  auto in_shape = module::getShape(getInput());
  int64_t shape_size = in_shape.size();
  p.dim = getAxis();
  p.N = in_shape[0];
  p.C = 1;
  p.H = 1;
  p.W = 1;
  switch (shape_size) {
  case 2:
    p.C = in_shape[1];
    break;
  case 3:
    p.C = in_shape[1];
    p.H = in_shape[2];
    break;
  case 4:
    p.C = in_shape[1];
    p.H = in_shape[2];
    p.W = in_shape[3];
  default:
    break;
  }
  p.dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::CumSumOp::get_fw_type_bm1684x() { return FW_BMNET_CUMSUM; }
