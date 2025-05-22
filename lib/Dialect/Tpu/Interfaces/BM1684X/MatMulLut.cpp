//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::MatMulLutOp::codegen_global_bm1684x() {
  auto p = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  int offset_table = 1;
  if (module::isMARS3()) {
    if (!p.hdim_is_batch) {
      BM168x::fix_shape(input_spec->at(0), {p.batch, p.M, p.K});
      if (p.right_transpose == false) {
        BM168x::fix_shape(input_spec->at(1 + offset_table),
                          {p.batch, p.K, p.N});
      } else {
        BM168x::fix_shape(input_spec->at(1 + offset_table),
                          {p.batch, p.N, p.K});
      }
      BM168x::fix_shape(output_spec->at(0), {p.batch, p.M, p.N});
    }
    lut_matmul_param_t spec{0};
    // LUT
    spec.table_length = 256;
    // Matmul
    spec.common_matmul_param.Y_dtype = output_spec->at(0).dtype;
    spec.common_matmul_param.L_trans = p.left_transpose;
    spec.common_matmul_param.R_trans = p.right_transpose;
    spec.common_matmul_param.has_bias = p.with_bias;
    spec.common_matmul_param.hdim_is_batch = p.hdim_is_batch;
    spec.common_matmul_param.requant_mode = -1;
    spec.common_matmul_param.do_relu = p.do_relu;
    spec.common_matmul_param.upper_limit = p.relu_limit;
    if (module::isUniformQuantized(getInput())) {
      spec.common_matmul_param.R_zp_is_const = true;
      spec.common_matmul_param.R_zp_const_val = p.right_zp;
      spec.common_matmul_param.izp_const_val = p.input_zp;
      if (module::isUniformQuantized(getOutput())) {
        spec.common_matmul_param.requant_mode =
            static_cast<int>(getQuantMode());
        auto rshift_v = module::getI64Array(getRshifts());
        auto multiplier_v = module::getI64Array(getMultipliers());
        spec.common_matmul_param.mul_val = multiplier_v->at(0);
        spec.common_matmul_param.shift_val = -rshift_v->at(0);
        auto output_type = module::getUniformQuantizedType(getOutput());
        spec.common_matmul_param.offset_val = output_type.getZeroPoint();
        spec.common_matmul_param.fuse_rq = getFuseRq();
        if (spec.common_matmul_param.fuse_rq)
          spec.common_matmul_param.round_mode = (RoundingMode)getRoundMode();
      }
    } else {
      assert(0);
    }

    BM168x::call_global_func("backend_api_matmul_lut_global", &spec,
                             sizeof(spec), input_spec->data(),
                             output_spec->data());
  } else {
    assert(0); // only  support LUT + MatMul, thus must be int8
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MatMulLutOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::MatMulLutOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
