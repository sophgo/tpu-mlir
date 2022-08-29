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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto in_type = Module::getStorageType(input());
  auto out_type = Module::getStorageType(output());
  bool isInQuant = Quant::isUniformQuantized(input());
  bool isOutQuant = Quant::isUniformQuantized(output());
  if (in_type.isF32() && out_type.isF16()) {
    f32_to_f16(p.inputs[0], p.outputs[0], num_elem);
  } else if (in_type.isF32() && out_type.isBF16()) {
    f32_to_bf16(p.inputs[0], p.outputs[0], num_elem);
  } else if (isOutQuant && false == isInQuant) {
    auto qtype = Quant::getUniformQuantizedType(output());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      float v =
          std::round(p.inputs[0][i] / qtype.getScale()) + qtype.getZeroPoint();
      if (out_type.isUnsignedInteger(8)) {
        p.outputs[0][i] = Quant::to_uint8(v);
      } else {
        p.outputs[0][i] = Quant::to_int8(v);
      }
    }
  } else if (isInQuant && false == isOutQuant) {
    auto qtype = Quant::getUniformQuantizedType(input());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] =
          qtype.getScale() * (p.inputs[0][i] - qtype.getZeroPoint());
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
