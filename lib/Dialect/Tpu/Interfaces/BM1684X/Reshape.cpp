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



using namespace tpu_mlir::backend;

void tpu::ReshapeOp::codegen_global_bm1684x() {
  // do nothing
}

//dynamic codegen
int64_t tpu::ReshapeOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(reshape_spec_t);
  reshape_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto out_shape = module::getShape(getOutput());
  spec.dims = out_shape.size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = out_shape[i];
  }

  auto p = static_cast<char *>(buffer);
  memcpy(p, &spec, sizeof(spec));
  p += sizeof(spec);
  return p - static_cast<char *>(buffer);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ReshapeOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(reshape_spec_t);
  reshape_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto out_shape = module::getShape(getOutput());
  spec.dims = out_shape.size();
  for (int i = 0; i < spec.dims; i++) {
    spec.shape[i] = out_shape[i];
  }

  auto p = static_cast<char *>(buffer);
  memcpy(p, &spec, sizeof(spec));
  p += sizeof(spec);
  return p - static_cast<char *>(buffer);
}
