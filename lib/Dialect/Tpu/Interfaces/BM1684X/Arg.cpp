//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::ArgOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  const bool need_val = !getValues().getType().isa<NoneType>();
  arg_global_spec_t spec = {0};
  spec.common.axis = getAxis();
  spec.common.method = StringSwitch<int>(getMode())
                           .Case("ArgMax", 0)
                           .Case("ArgMin", 1)
                           .Default(-1);
  spec.common.need_val = need_val;
  spec.common.is_index_int32 = true;

  BM168x::call_global_func("backend_api_arg_global", &spec,
                           sizeof(arg_global_spec_t), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ArgOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }
