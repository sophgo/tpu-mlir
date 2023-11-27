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
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MulConstOp::init(InferenceParameter &p) { return success(); }

void tpu::MulConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] * getConstVal().convertToDouble();
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(),
                                              getRshift());
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.inputs[0][i];
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    }
  }
  return success();
}

LogicalResult tpu::MulConstOp::LocalGenSupport() { return success(); }

void tpu::MulConstOp::assign_fw_param(void *param) {
  IR_PARAM_CONST_BINARY(BINARY_MUL);
}

ArrayAttr tpu::MulConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

// case1: Fuse multiple mulconst ops into one
// only when in_dtype == out_dtype or in_dtype == fp8
struct FuseMultiMulConst : public OpRewritePattern<tpu::MulConstOp> {
  FuseMultiMulConst(mlir::MLIRContext *context)
      : OpRewritePattern<tpu::MulConstOp>(context) {}
  LogicalResult
  matchAndRewrite(tpu::MulConstOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // starts from the last mulconst op
    if (!op->hasOneUse() ||
        dyn_cast<tpu::MulConstOp>(module::getNextOp(op, 0))) {
      return failure();
    }

    auto input = op.getInput();
    auto in_dtype = BM168x::getDataType(input);
    auto out_dtype = BM168x::getDataType(op.getOutput());
    auto final_const_val = op.getConstVal().convertToDouble();
    auto prev_op = dyn_cast<tpu::MulConstOp>(input.getDefiningOp());
    if (!prev_op) {
      return failure();
    }

    while (in_dtype == out_dtype || in_dtype == DTYPE_F8E4M3 ||
           in_dtype == DTYPE_F8E5M2) {
      final_const_val *= prev_op.getConstVal().convertToDouble();
      input = prev_op.getInput();
      prev_op = dyn_cast<tpu::MulConstOp>(input.getDefiningOp());
      if (!prev_op) {
        break;
      }
      out_dtype = in_dtype;
      in_dtype = BM168x::getDataType(prev_op.getInput());
    }
    op.setConstValAttr(rewriter.getF64FloatAttr(final_const_val));
    op.setOperand(input);

    return success();
  }
};

// case2: Fuse cast to FP8 MulConst
struct FuseCastToF8MulConst : public OpRewritePattern<tpu::MulConstOp> {
  FuseCastToF8MulConst(mlir::MLIRContext *context)
      : OpRewritePattern<tpu::MulConstOp>(context) {}
  LogicalResult
  matchAndRewrite(tpu::MulConstOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto in_dtype = BM168x::getDataType(input);
    if (!(in_dtype == DTYPE_F8E4M3 || in_dtype == DTYPE_F8E5M2)) {
      return failure();
    }

    if (!op->hasOneUse()) {
      return failure();
    }

    auto castOp = dyn_cast<tpu::CastOp>(module::getNextOp(op, 0));
    if (!castOp) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<tpu::MulConstOp>(
        castOp, castOp.getType(), ValueRange{input}, op->getAttrs());
    return success();
  }
};

void tpu::MulConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<FuseMultiMulConst, FuseCastToF8MulConst>(context);
}
