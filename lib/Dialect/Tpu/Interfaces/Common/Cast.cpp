//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

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
      p_dst[i] = bf16_mul(BF16(p_src[i], false), scale);
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
        v = bf16_mul(BF16(p.inputs[0][i], false), BF16(1. / qtype.getScale()));
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
  } else if (in_type.isF32() && out_type.isInteger(32)){
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = round(p.inputs[0][i]);
    }
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
    if (type_need_cast(in_type, out_type) == false) {
      // for example, int32 cast int32 => remove this one cast
      rewriter.replaceOp(op, {in});
      return success();
    }
    auto castInputOp = dyn_cast<tpu::CastOp>(module::getOriValue(in).getDefiningOp());
    if (!castInputOp || castInputOp->hasOneUse() == false) {
      return failure();
    }
    auto pre_in = castInputOp.getInput();
    if (out_type == pre_in.getType()) {
      // for example, int32 cast f16 cast int32 => remove these two cast
      rewriter.replaceOp(op, {pre_in});
      return success();
    }
    bool is_qtype_out = module::isUniformQuantized(out_type);
    bool is_qtype_in = module::isUniformQuantized(in);
    bool is_qtype_pre_in = module::isUniformQuantized(pre_in);
    if (false == is_qtype_out && false == is_qtype_pre_in) {
      // for example, int32 cast int8 cast f16 => int32 cast f16
      op->setOperand(0, pre_in);
      rewriter.eraseOp(castInputOp);
      return success();
    }
    if (is_qtype_out && false == is_qtype_in && false == is_qtype_pre_in) {
      auto pre_stype = module::getStorageType(pre_in);
      if (pre_stype.isa<mlir::FloatType>()) {
        // for example, f32 cast f16, f16 cast int8 => f32 cast int8
        op->setOperand(0, pre_in);
        rewriter.eraseOp(castInputOp);
        return success();
      }
      if (pre_stype.isIntOrIndex()) {
        // for example, int32 cast f32, f32 cast int8 => int32 requant to int8
        auto qtype = module::getUniformQuantizedType(out_type);
        int32_t multiplier;
        int32_t shift;
        std::vector<NamedAttribute> attrs;
        get_scale_and_shift(1.0 / qtype.getScale(), multiplier, shift, 32);
        auto ctx = op.getContext();
        attrs.push_back(rewriter.getNamedAttr(
            "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
        attrs.push_back(rewriter.getNamedAttr(
            "rshift", rewriter.getSI32IntegerAttr(shift)));
        attrs.push_back(rewriter.getNamedAttr(
            "quant_mode",
            tpu::RequantModeAttr::get(ctx, tpu::RequantMode::MultiplierShift)));
        rewriter.replaceOpWithNewOp<tpu::RequantIntOp>(
            op, op.getOutput().getType(), ValueRange{pre_in}, attrs);
        return success();
      }
    }
    castInputOp.dump();
    op.dump();
    llvm::errs() << "Warning: two cast can merge to one !!!\n";
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
  if (module::isBM1684Family()) {
    auto in_dtype = BM168x::getDataType(getInput());
    if (in_dtype == DTYPE_INT32) {
      return failure();
    }
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
