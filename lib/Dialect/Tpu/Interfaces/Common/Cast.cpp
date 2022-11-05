//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

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

float dequant(const float &data, const quant::UniformQuantizedType &qtype) {
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

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto in_type = Module::getStorageType(input());
  auto out_type = Module::getStorageType(output());
  bool isInQuant = Quant::isUniformQuantized(input());
  bool isOutQuant = Quant::isUniformQuantized(output());
  auto op = getOperation();
  auto chip = Module::getChip(op);
  bool is_cv18xx = Module::isCV18xx(chip);
  auto round_mode = is_cv18xx ? ROUNDING_HALF_TO_EVEN : ROUNDING_HALF_DOWN;
  bool is_tpu = Module::isTpuOp(op);

  if (in_type.isF32() && out_type.isF16()) {
    f32_to_f16(p.inputs[0], p.outputs[0], num_elem);
  } else if (in_type.isF32() && out_type.isBF16()) {
    f32_to_bf16(p.inputs[0], p.outputs[0], num_elem, is_cv18xx, false);
  } else if (isOutQuant && false == isInQuant) {
    // FP32|BF16|F16|... => INT8|UINT8|...
    auto qtype = Quant::getUniformQuantizedType(output());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      float v;
      if (is_cv18xx) {
        v = cvi_f32_to_fbf16(
            cvi_f32_to_fbf16(cvi_f32_to_fbf16(p.inputs[0][i], false) *
                             cvi_f32_to_fbf16(1. / qtype.getScale())) +
                cvi_f32_to_fbf16(qtype.getZeroPoint()),
            (qtype.getZeroPoint() != 0));
      } else {
        v = requant(p.inputs[0][i], qtype);
      }
      if (out_type.isUnsignedInteger(8)) {
        p.outputs[0][i] = Quant::to_uint8(v, round_mode);
      } else {
        p.outputs[0][i] = Quant::to_int8(v, round_mode);
      }
    }
  } else if (isInQuant && false == isOutQuant) {
    // INT8|UINT8|... ==> FP32|BF16|F16|...
    auto qtype = Quant::getUniformQuantizedType(input());
    if (is_cv18xx) {
      cvi_int8_to_bf16(p.inputs[0], p.outputs[0], qtype.getScale(),
                       -qtype.getZeroPoint(), num_elem, is_tpu);
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (size_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = dequant(p.inputs[0][i], qtype);
      }
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
    auto in = op.input();
    auto in_type = in.getType();
    auto out_type = op.output().getType();
    if (in_type == out_type) {
      rewriter.replaceOp(op, {in});
      return success();
    }
    auto castInputOp = in.getDefiningOp<tpu::CastOp>();
    if (!castInputOp) {
      return failure();
    }

    if (out_type == castInputOp.input().getType()) {
      rewriter.replaceOp(op, {castInputOp.input()});
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
