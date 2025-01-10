//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================

void tpu::MaxPoolingIndicesBwdOp::codegen_global_bm1684x() {
  maxpooling_indices_bwd_spec_t spec;
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  memset(&spec, 0, sizeof(spec));
  auto kernel = module::getI64Array(getKernelShape());
  auto strides = module::getI64Array(getStrides());
  auto paddings = module::getI64Array(getPads());
  spec.kernels[0] = kernel->at(0);
  spec.kernels[1] = kernel->at(1);
  spec.strides[0] = strides->at(0);
  spec.strides[1] = strides->at(1);
  spec.paddings[0] = paddings->at(0);
  spec.paddings[1] = paddings->at(1);
  BM168x::call_global_func("backend_api_maxpooling_indices_bwd_global", &spec,
                           sizeof(spec), input_spec->data(),
                           output_spec->data());
}

void tpu::MaxPoolingIndicesBwdOp::codegen_global_bm1684() { return; }

void tpu::MaxPoolingIndicesBwdOp::codegen_global_cv18xx(int64_t layer_id) {}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
uint32_t tpu::MaxPoolingIndicesBwdOp::dyn_codegen_global_bm1684(void *buffer) {
  return -1;
}

int64_t tpu::MaxPoolingIndicesBwdOp::get_fw_type_bm1684() { return 0; }

int64_t tpu::MaxPoolingIndicesBwdOp::dyn_codegen_global_bm1684x(void *buffer) {
  return -1;
}

int64_t tpu::MaxPoolingIndicesBwdOp::get_fw_type_bm1684x() { return 0; }