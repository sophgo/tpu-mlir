//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

float requant(const float &data, const quant::UniformQuantizedType &qtype) {
  auto stype = qtype.getExpressedType();
  if (stype.isF32()) {
    return std::round(data * (float)(1.0 / qtype.getScale())) +
           qtype.getZeroPoint();
  }
  if (stype.isF16()) {
    return std::round(F16(data * F16(1.0 / qtype.getScale()))) +
           qtype.getZeroPoint();
  }
  if (stype.isBF16()) {
    return std::round(BF16(data * BF16(1.0 / qtype.getScale()))) +
           qtype.getZeroPoint();
  }
  qtype.dump();
  llvm_unreachable("Unsupport type");
}

static float dequant(const float &data,
                     const quant::UniformQuantizedType &qtype) {
  auto stype = qtype.getExpressedType();
  if (stype.isF32()) {
    return (float)qtype.getScale() * (data - (float)qtype.getZeroPoint());
  }
  if (stype.isF16()) {
    return F16(F16(qtype.getScale()) * F16(data - (float)qtype.getZeroPoint()));
  }
  if (stype.isBF16()) {
    return BF16(BF16(qtype.getScale()) *
                BF16(data - (float)qtype.getZeroPoint()));
  }
  qtype.dump();
  llvm_unreachable("Unsupport type");
}

static void cvi_int8_to_bf16(float *p_src, float *p_dst, float scale, int num,
                             bool is_tpu) {
  // int8 / uint8 ==> bf16 / fp32
  if (is_tpu) {
    scale = BF16(scale);
#pragma omp parallel for schedule(static, omp_schedule(num))
    for (int i = 0; i < num; i++) {
      p_dst[i] = BF16(BF16(p_src[i]) * scale);
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num))
    for (int i = 0; i < num; i++) {
      p_dst[i] = p_src[i] * scale;
    }
  }
}

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool isInQuant = module::isUniformQuantized(getInput());
  bool isOutQuant = module::isUniformQuantized(getOutput());
  auto op = getOperation();
  bool is_cv18xx = module::isCV18xx();
  auto round_mode =
      is_cv18xx ? ROUNDING_HALF_TO_EVEN : ROUNDING_HALF_AWAY_FROM_ZERO;
  bool is_tpu = module::isTpuOp(op);

  if (in_type.isF32() && out_type.isF16()) {
    F16(p.inputs[0], p.outputs[0], num_elem);
  } else if (in_type.isF32() && out_type.isBF16()) {
    BF16(p.inputs[0], p.outputs[0], num_elem, false);
  } else if (isOutQuant && false == isInQuant) {
    // FP32|BF16|F16|... => INT8|UINT8|...
    auto qtype = module::getUniformQuantizedType(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      float v;
      if (is_cv18xx) {
        v = BF16(BF16(p.inputs[0][i], false) * BF16(1. / qtype.getScale()));
      } else {
        v = requant(p.inputs[0][i], qtype);
      }
      p.outputs[0][i] = saturate(v, out_type, round_mode);
    }
  } else if (isInQuant && false == isOutQuant) {
    // INT8|UINT8|... ==> FP32|BF16|F16|...
    auto qtype = module::getUniformQuantizedType(getInput());
    if (is_cv18xx) {
      cvi_int8_to_bf16(p.inputs[0], p.outputs[0], qtype.getScale(), num_elem,
                       is_tpu);
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = dequant(p.inputs[0][i], qtype);
      }
    }
    //   } else if (isInQuant && isOutQuant)  {
    //     auto in_qtype = module::getUniformQuantizedType(getInput());
    //     auto out_qtype = module::getUniformQuantizedType(getOutput());
    //     if (in_qtype.getScale() == out_qtype.getScale() &&
    //         in_type.isInteger(8) && out_type.isInteger(8)) {
    //       int zero_diff = in_qtype.getZeroPoint() - out_qtype.getZeroPoint();
    //       if (zero_diff == 0) {
    //         std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
    //       } else {
    // #pragma omp parallel for schedule(static, omp_schedule(num_elem))
    //         for (int64_t i = 0; i < num_elem; i++) {
    //           p.outputs[0][i] = (p.inputs[0][i] - zero_diff);
    //         }
    //       }
    //     } else {
    //       int64_t multi, shift_val;
    //       QuantizeMultiplier(in_qtype.getScale() / out_qtype.getScale(),
    //       &multi, &shift_val); for (int64_t i = 0; i < num_elem; ++i) {
    //         auto v = out_qtype.getZeroPoint() +
    //         MultiplyByQuantizedMultiplier(
    //                                     (int32_t)(p.inputs[0][i]) -
    //                                     in_qtype.getZeroPoint(),
    //                                     (int32_t)multi, (int32_t)shift_val);
    //         p.outputs[0][i] = saturate(v, out_type);
    //       }
    //     }
  } else {
    std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
  }

  return success();
}

struct SimplifyRedundantCast : public OpRewritePattern<tpu::CastOp> {
  SimplifyRedundantCast(mlir::MLIRContext *context)
      : OpRewritePattern<tpu::CastOp>(context, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(tpu::CastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto in_type = in.getType();
    auto out_type = op.getOutput().getType();
    if (in_type == out_type) {
      rewriter.replaceOp(op, {in});
      return success();
    }
    auto castInputOp = in.getDefiningOp<tpu::CastOp>();
    if (!castInputOp) {
      return failure();
    }

    if (out_type == castInputOp.getInput().getType()) {
      rewriter.replaceOp(op, {castInputOp.getInput()});
      return success();
    }
    return failure();
  }
};

void tpu::CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SimplifyRedundantCast>(context);
}

mlir::Type tpu::CastOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return do_nothing(mode);
}

LogicalResult tpu::CastOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    auto in_type = module::getStorageType(getInput());
    auto out_type = module::getStorageType(getOutput());
    // type.isSignedInteger()
    if ((in_type.getIntOrFloatBitWidth() == 8 && out_type.isBF16()) ||
        (in_type.isBF16() && out_type.isSignedInteger())) {
      return success();
    }
    return failure();
  }
  return success();
}

void tpu::CastOp::assign_fw_param(void *param) {
  fw_dtype_convert_layer_param_t fw_param = {0};
  fw_param.src_type = BM168x::getDataType(getInput());
  fw_param.dst_type = BM168x::getDataType(getOutput());
  fw_param.src_stmode = BM1684::getStoreMode(getInput());
  fw_param.dst_stmode = BM1684::getStoreMode(getOutput());
  fw_param.round_mode = ROUND_INF; // if support other round_mode, change here
  memcpy(param, &fw_param, sizeof(fw_dtype_convert_layer_param_t));
}
