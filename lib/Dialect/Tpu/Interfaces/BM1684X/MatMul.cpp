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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/WeightReorder.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
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
  // if (!Module::getStorageType(op.bias()).isInteger(32))
  //   return failure();

  int64_t batch, M, K, N, right_zp, input_zp;
  bool with_bias, relu, right_transpose;
  double relu_limit;
  op.parseParam(batch, M, K, N, with_bias, relu, relu_limit, right_zp,
             right_transpose, input_zp);

  // bias merge input zp
  if (input_zp == 0)
    return failure();
  std::shared_ptr<std::vector<int32_t>> bias_quant;
  if (isa<top::WeightOp>(op.bias().getDefiningOp())) {
    bias_quant = cast<top::WeightOp>(op.bias().getDefiningOp()).read<int32_t>();
    for (size_t i = 0; i < N; ++i) {
      bias_quant->data()[i] += input_zp * right_zp * K;
    }
  } else {
    bias_quant = std::shared_ptr<std::vector<int32_t>>(
        new std::vector<int32_t>(N, 0));
    for (size_t i = 0; i < N; ++i) {
      bias_quant->data()[i] += input_zp * right_zp * K;
    }
    auto stype = Module::getStorageType(op.bias());
    // std::vector<int64_t> bias_shape = {N};
    auto new_type = RankedTensorType::get({N}, rewriter.getI32Type());
    auto new_op =
        top::WeightOp::create(op, "bias_merge_izp", *bias_quant, new_type);
    op->setOperand(2, new_op);
  }
  return success();
}

void tpu::MatMulOp::codegen_global_bm1684x() {
  int64_t batch, M, K, N, right_zp, input_zp;
  bool with_bias, relu, right_transpose;
  double relu_limit;
  parseParam(batch, M, K, N, with_bias, relu, relu_limit, right_zp,
             right_transpose, input_zp);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (batch != 1) {
    BM168x::fix_shape(input_spec->at(0),
                       {(int32_t)batch, (int32_t)M, (int32_t)K});
    BM168x::fix_shape(input_spec->at(1),
                       {(int32_t)batch, (int32_t)K, (int32_t)N});
    BM168x::fix_shape(output_spec->at(0),
                       {(int32_t)batch, (int32_t)M, (int32_t)N});
    batch_matmul_common_spec_t spec{0};
    spec.Y_dtype = output_spec->at(0).dtype;
    spec.L_trans = false;
    spec.R_trans = right_transpose;
    spec.has_bias = with_bias;
    spec.hdim_is_batch = false;
    spec.requant_mode = -1;
    if (Quant::isUniformQuantized(input())) {
      spec.R_zp_is_const = true;
      spec.R_zp_const_val = right_zp;
      spec.izp_const_val = input_zp;
      if (Quant::isUniformQuantized(output())) {
        spec.requant_mode = static_cast<int>(quant_mode());
        auto rshift_v = Module::getI64Array(rshifts(), 1, 0);
        auto multiplier_v = Module::getI64Array(multipliers(), 1, 1);
        assert(rshift_v->size() == 1);
        assert(multiplier_v->size() == 1);
        spec.mul_val = multiplier_v->at(0);
        spec.shift_val = -rshift_v->at(0);
        auto output_type = Quant::getUniformQuantizedType(output());
        spec.offset_val = output_type.getZeroPoint();
      }
    }

    BM168x::call_global_func(
        "backend_api_batch_matmul_global", &spec, sizeof(spec),
        input_spec->data(), output_spec->data());
    return;
  }
  BM168x::fix_shape(input_spec->at(0), {(int32_t)M, (int32_t)K});
  BM168x::fix_shape(input_spec->at(1), {(int32_t)K, (int32_t)N});
  BM168x::fix_shape(output_spec->at(0), {(int32_t)M, (int32_t)N});
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.if_relu = relu;
  spec.relu_limit = relu_limit;
  spec.have_bias = with_bias;
  spec.requant_mode = -1;
  spec.R_transpose = right_transpose;
  if (Quant::isUniformQuantized(input())) {
    spec.rshift = 0;
    spec.is_asymmetric = 1;
    spec.rzp_is_const = 1;
    spec.rzp_const_val = right_zp;
    spec.izp_const_val = input_zp;
    if (Quant::isUniformQuantized(output())) {
      auto rshift_v = Module::getI64Array(rshifts(), 1, 0);
      auto multiplier_v = Module::getI64Array(multipliers(), 1, 1);
      assert(rshift_v->size() == 1);
      assert(multiplier_v->size() == 1);
      spec.requant_mode = static_cast<int>(quant_mode());
      spec.mul_val = multiplier_v->at(0);
      spec.shift_val = -rshift_v->at(0);
      auto output_type = Quant::getUniformQuantizedType(output());
      spec.offset_val = output_type.getZeroPoint();
      spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
    }
  }
  BM168x::call_global_func("backend_api_fc_global", &spec,
                                       sizeof(spec), input_spec->data(),
                                       output_spec->data());
}
