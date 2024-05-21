//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"

#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;
namespace tpu_mlir {

template <typename OpTy>
class TopLoweringToTosa : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {
    Lowering(rewriter, opTy);
    return success();
  }

  virtual void Lowering(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
};

// NCHW -> NHWC
static Type change_dataformat(Type ty_){
  auto ty = ty_.cast<RankedTensorType>();
  if (ty.getShape().size() != 4) return ty;
  auto n = ty.getShape()[0]; // N
  auto h = ty.getShape()[2]; // H
  auto w = ty.getShape()[3]; // W
  auto c = ty.getShape()[1]; // C
  std::vector<int64_t> newShape{n, h, w, c};
  return RankedTensorType::get(newShape, ty.getElementType());
}


static float* change_weight(std::shared_ptr<std::vector<float>> valptr,
                                    Type ty_) {
  auto ty = ty_.cast<RankedTensorType>();
  if (ty.getShape().size() != 4) return valptr->data();
  auto n = ty.getShape()[0];
  auto h = ty.getShape()[2];
  auto w = ty.getShape()[3];
  auto c = ty.getShape()[1];
  float* new_val = new float[valptr->size()];
  int dst, src, ds_1, d_2, d_3, s_3;
  int a_ds = h*w*c, b_d = w*c, b_s = h*w;
  for (int i = 0; i < n; i++) {
    ds_1 = i * a_ds;
    for (int j = 0; j < h; j++) {
      d_2 = j * b_d;
      s_3 = j * w;
      for (int k = 0; k < w; k++) {
        d_3 = k * c;
        for (int p = 0; p < c; p++){
          dst = ds_1 + d_2   + d_3 + p;    // nhwc
          src = ds_1 + p*b_s + s_3 + k;    // nchw
          new_val[dst] = valptr->data()[src];
        }
      }
    }
  }
  return new_val;
}


static std::vector<NamedAttribute> gen_clamp_attr(PatternRewriter &rewriter,
                      Type newType, ::llvm::APFloat relu_limit) {
  std::vector<NamedAttribute> clamp_attr;
  clamp_attr.push_back(rewriter.getNamedAttr("min_int", rewriter.getI64IntegerAttr(0)));
  clamp_attr.push_back(rewriter.getNamedAttr("max_int", rewriter.getI64IntegerAttr(0)));
  clamp_attr.push_back(rewriter.getNamedAttr("min_fp", rewriter.getF32FloatAttr(0)));
  auto floatType = newType.cast<RankedTensorType>().getElementType().cast<FloatType>();
  const llvm::fltSemantics &semantic = floatType.getFloatSemantics();
  auto zero = llvm::APFloat::getZero(relu_limit.getSemantics());   // Negative = false
  if (relu_limit < zero) {
    clamp_attr.push_back(rewriter.getNamedAttr("max_fp",
        rewriter.getFloatAttr(floatType, APFloat::getInf(semantic)))); // Negative = false
  } else {
    clamp_attr.push_back(rewriter.getNamedAttr("max_fp",
        rewriter.getFloatAttr(floatType, relu_limit)));
  }
  return clamp_attr;
}

} // namespace tpu_mlir
