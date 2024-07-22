//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Traits/Traits.h"
#include <llvm/Support/Debug.h>
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {
// ===================================
// helper functions
// ===================================
void dump(Operation *op);

Value getTheOtherOperand(Operation *op, Value curr);

void distribute(PatternRewriter &rewriter, std::vector<Operation *> ops_begin,
                std::vector<Operation *> ops_end,
                tpu::DevPattern pattern,
                std::vector<int64_t> &begin_methods,
                std::vector<int64_t> &end_methods,
                int num_head);
void distribute(PatternRewriter &rewriter, Operation *op_begin,
                Operation *op_end, tpu::DevPattern pattern);
void distributeAfter(PatternRewriter &rewriter, Operation *op_begin,
                     Operation *op_end, tpu::DevPattern pattern);
bool isLargeMatMul(Operation *op);
bool isBinaryOp(Operation *op);
Operation *cloneOp(PatternRewriter &rewriter, Operation *op,
                   llvm::ArrayRef<int64_t> new_shape, llvm::StringRef suffix);
// erase op and its uers
void eraseForward(PatternRewriter &rewriter, Operation *op);

// ===================================
// patterns for distribution
// ===================================
template <typename MatMulTy>
class  MatMulSliceMerge : public OpRewriterPatternEx<MatMulTy> {
public:
   MatMulSliceMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulTy>(context,"MatMulSliceMerge") {}

  LogicalResult matchAndRewriteImpl(MatMulTy op,
                                    PatternRewriter &rewriter) const override ;
};

template <typename MatMulTy>
class  AttentionSliceMerge : public OpRewriterPatternEx<MatMulTy> {
public:
   AttentionSliceMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulTy>(context,"AttentionSliceMerge") {}

  LogicalResult matchAndRewriteImpl(MatMulTy op,
                                    PatternRewriter &rewriter) const override ;
};

template <typename MatMulTy>
class  MatMulSliceMerge2 : public OpRewriterPatternEx<MatMulTy> {
public:
   MatMulSliceMerge2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulTy>(context,"MatMulSliceMerge2") {}

  LogicalResult matchAndRewriteImpl(MatMulTy op,
                                    PatternRewriter &rewriter) const override ;
};

template <typename MatMulTy>
class  AttentionSliceMerge2 : public OpRewriterPatternEx<MatMulTy> {
public:
   AttentionSliceMerge2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulTy>(context,"AttentionSliceMerge2") {}

  LogicalResult matchAndRewriteImpl(MatMulTy op,
                                    PatternRewriter &rewriter) const override ;
};


class  MatMulSliceMerge3 : public OpRewriterPatternEx<tpu::AddOp> {
public:
   MatMulSliceMerge3(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::AddOp>(context,"MatMulSliceMerge3") {}

  LogicalResult matchAndRewriteImpl(tpu::AddOp op,
                                    PatternRewriter &rewriter) const override ;
};
template <typename MatMulTy>
class  MatMulTopK : public OpRewriterPatternEx<MatMulTy> {
public:
   MatMulTopK(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulTy>(context,"MatMulTopK") {}

  LogicalResult matchAndRewriteImpl(MatMulTy op,
                                    PatternRewriter &rewriter) const override ;
};
class  EmbeddingSliceMerge : public OpRewriterPatternEx<tpu::GatherOp> {
public:
  EmbeddingSliceMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::GatherOp>(context,"EmbeddingSliceMerge") {}

  LogicalResult matchAndRewriteImpl(tpu::GatherOp op,
                                    PatternRewriter &rewriter) const override ;
};

class  FAttentionSliceMerge : public OpRewriterPatternEx<tpu::FAttentionOp> {
public:
  FAttentionSliceMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::FAttentionOp>(context,"FAttentionSliceMerge") {}

  LogicalResult matchAndRewriteImpl(tpu::FAttentionOp op,
                                    PatternRewriter &rewriter) const override ;
};

template <typename MatMulTy>
void sliceMergeSplit(MatMulTy mm0, PatternRewriter &rewriter,
                     tpu::DevBeginOp op, int64_t num_devices);

void sliceAttentionMergeSplit(PatternRewriter &rewriter, tpu::DevBeginOp op,
                              int64_t num_devices);

template <typename MatMulTy>
void sliceAttentionMergeSplit(PatternRewriter &rewriter, tpu::DevBeginOp op,
                              int64_t num_devices);

void sliceMerge2Split(PatternRewriter &rewriter, tpu::DevBeginOp op,
                      int64_t num_devices);

void sliceAttentionMerge2Split(PatternRewriter &rewriter,
                               tpu::DevBeginOp op,
                               int64_t num_devices);

template <typename MatMulTy>
void sliceAttentionMerge2Split(PatternRewriter &rewriter,
                               tpu::DevBeginOp op,
                               int64_t num_devices);

void sliceMerge3Split(PatternRewriter &rewriter, tpu::DevBeginOp op,
                      int64_t num_devices);

template <typename MatMulTy>
void topKSplit(MatMulTy mm, PatternRewriter &rewriter,
               tpu::DevBeginOp op, int64_t num_devices);

void embeddingMergeSplit(PatternRewriter &rewriter, tpu::DevBeginOp op,
                    int64_t num_devices);

void sliceFAttentionMergeSplit(PatternRewriter &rewriter,
                               tpu::DevBeginOp op,
                               int64_t num_devices);

enum class DevEndMode {
  EndToUnknown = 0,
  EndToSum = 1,
  EndToTopK = 2,
  EndToConcat = 3
};


enum class DevBeginMethod {
  BeginFromUnknown = 0,
  BeginFromCopy = 1,
  BeginFromSplit = 2,
};

enum class DevEndMethod {
  EndToUnknown = 0,
  EndToSum = 1,
  EndToTopK = 2,
  EndToConcat = 3,
};

typedef std::shared_ptr<std::vector<int64_t>> begin_method_array_t;
typedef std::shared_ptr<std::vector<int64_t>> end_method_array_t;

static begin_method_array_t getBeginMethodArray(tpu::DevBeginOp op) {
  auto begin_methods = op.getBeginMethods();
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto it : llvm::enumerate(begin_methods)) {
    auto attr = it.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      llvm_unreachable("not DevBeginMethod type");
    }
  }
  return std::move(data);
}

static end_method_array_t getEndMethodArray(tpu::DevEndOp op) {
  auto end_methods = op.getEndMethods();
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto it : llvm::enumerate(end_methods)) {
    auto attr = it.value().dyn_cast<IntegerAttr>();
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      llvm_unreachable("not DevEndMethod type");
    }
  }
  return std::move(data);
}

static DevEndMode getEndMode(tpu::DevEndOp op) {
  switch (op.getPattern()) {
  case tpu::DevPattern::MatMulSliceMerge:
  case tpu::DevPattern::MatMulSliceMerge2:
  case tpu::DevPattern::MatMulSliceMerge3:
    return DevEndMode::EndToSum;
  case tpu::DevPattern::MatMulTopK:
    return DevEndMode::EndToTopK;
  default:
    UNREACHABLE_OP("Not Implemented", op);
  }
  return DevEndMode::EndToUnknown;
}

// ===================================
// split module to multi modules for devices
// ===================================
void distributeModules(ModuleOp module, int64_t num_device);

} // namespace tpu
} // namespace tpu_mlir
