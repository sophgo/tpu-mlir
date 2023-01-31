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
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"

using namespace tpu_mlir::backend;
using namespace tpu_mlir::bm1684x;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fc_global_spec {
  /* common param of float and fixed */
  int32_t R_transpose;
  int32_t have_bias;
  int32_t if_relu;
  float relu_limit;
  /* quantize param */
  int32_t rshift;
  int32_t is_asymmetric;
  int32_t rzp_const_val;
  int32_t rzp_is_const;
  int32_t izp_const_val;
  /* requantize param */
  int32_t requant_mode; // mode < 0 means no requantize
  int32_t mul_val;
  int32_t shift_val;
  int32_t offset_val;
  int32_t round_mode;
} fc_global_spec_t;

typedef struct batch_matmul_common_spec {
  int Y_dtype;
  int L_trans;
  int R_trans;
  bool R_zp_is_const;
  int R_zp_const_val;
  int izp_const_val;
  bool has_bias;
  bool hdim_is_batch;
  /* requant param */
  int requant_mode; // mode < 0 means no requantize
  int mul_val;
  int shift_val;
  int offset_val;
  // int round_mode;
} batch_matmul_common_spec_t;

#ifdef __cplusplus
}
#endif

template <>
LogicalResult WeightReorder<tpu::MatMulOp, int8_t>::matchAndRewrite(
    tpu::MatMulOp op, PatternRewriter &rewriter) const {
  // if (!module::getStorageType(op.getBias()).isInteger(32))
  //   return failure();
  auto p = op.parseParam();

  // bias merge input zp
  if (p.input_zp == 0)
    return failure();
  i32_array_t bias_quant;
  if (isa<top::WeightOp>(op.getBias().getDefiningOp())) {
    bias_quant =
        cast<top::WeightOp>(op.getBias().getDefiningOp()).read<int32_t>();
    for (size_t i = 0; i < p.N; ++i) {
      bias_quant->data()[i] += p.input_zp * p.right_zp * p.K;
    }
  } else {
    bias_quant = i32_array_t(new std::vector<int32_t>(p.N, 0));
    for (size_t i = 0; i < p.N; ++i) {
      bias_quant->data()[i] += p.input_zp * p.right_zp * p.K;
    }
    auto stype = module::getStorageType(op.getBias());
    // std::vector<int64_t> bias_shape = {N};
    auto new_type = RankedTensorType::get({p.N}, rewriter.getI32Type());
    auto new_op =
        top::WeightOp::create(op, "bias_merge_izp", *bias_quant, new_type);
    op->setOperand(2, new_op);
  }
  return success();
}

void tpu::MatMulOp::codegen_global_bm1684x() {
  auto p = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (p.batch != 1) {
    BM168x::fix_shape(input_spec->at(0), {p.batch, p.M, p.K});
    BM168x::fix_shape(input_spec->at(1), {p.batch, p.K, p.N});
    BM168x::fix_shape(output_spec->at(0), {p.batch, p.M, p.N});
    batch_matmul_common_spec_t spec{0};
    spec.Y_dtype = output_spec->at(0).dtype;
    spec.L_trans = false;
    spec.R_trans = p.right_transpose;
    spec.has_bias = p.with_bias;
    spec.hdim_is_batch = false;
    spec.requant_mode = -1;
    if (module::isUniformQuantized(getInput())) {
      spec.R_zp_is_const = true;
      spec.R_zp_const_val = p.right_zp;
      spec.izp_const_val = p.input_zp;
      if (module::isUniformQuantized(getOutput())) {
        spec.requant_mode = static_cast<int>(getQuantMode());
        auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
        auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
        assert(rshift_v->size() == 1);
        assert(multiplier_v->size() == 1);
        spec.mul_val = multiplier_v->at(0);
        spec.shift_val = -rshift_v->at(0);
        auto output_type = module::getUniformQuantizedType(getOutput());
        spec.offset_val = output_type.getZeroPoint();
      }
    }

    BM168x::call_global_func("backend_api_batch_matmul_global", &spec,
                             sizeof(spec), input_spec->data(),
                             output_spec->data());
    return;
  }
  BM168x::fix_shape(input_spec->at(0), {p.M, p.K});
  BM168x::fix_shape(input_spec->at(1), {p.K, p.N});
  BM168x::fix_shape(output_spec->at(0), {p.M, p.N});
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.if_relu = p.do_relu;
  spec.relu_limit = p.relu_limit;
  spec.have_bias = p.with_bias;
  spec.requant_mode = -1;
  spec.R_transpose = p.right_transpose;
  if (module::isUniformQuantized(getInput())) {
    spec.rshift = 0;
    spec.is_asymmetric = 1;
    spec.rzp_is_const = 1;
    spec.rzp_const_val = p.right_zp;
    spec.izp_const_val = p.input_zp;
    if (module::isUniformQuantized(getOutput())) {
      auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
      auto multiplier_v = module::getI64Array(getMultipliers(), 1, 1);
      assert(rshift_v->size() == 1);
      assert(multiplier_v->size() == 1);
      spec.requant_mode = static_cast<int>(getQuantMode());
      spec.mul_val = multiplier_v->at(0);
      spec.shift_val = -rshift_v->at(0);
      auto output_type = module::getUniformQuantizedType(getOutput());
      spec.offset_val = output_type.getZeroPoint();
      spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
    }
  }
  BM168x::call_global_func("backend_api_fc_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MatMulOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }
