//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/Patterns.h"

namespace tpu_mlir {
namespace tpu {

namespace tpu_dialect_tiling_primitives {
inline Value get_unique_operand_or_null(Operation *op) {
  Value result;
  int count = 0;
  for (auto operand : op->getOperands()) {
    if (isa<BlockArgument>(operand))
      continue;
    if (isa<top::WeightOp, top::NoneOp>(operand.getDefiningOp()))
      continue;
    result = operand;
    count++;
  }
  if (count > 1) {
    llvm::errs() << "Warning: this api is designed for ONE-to-ONE op chain.\n";
  }
  return count == 1 ? result : nullptr;
}

template <typename TargetOpType>
TargetOpType
get_prev_op(Value val, const std::initializer_list<mlir::TypeID> &allowed_ops) {
  if (!val || isa<BlockArgument>(val))
    return nullptr;
  auto def_op = val.getDefiningOp();
  if (auto target = dyn_cast<TargetOpType>(def_op)) {
    return target;
  }
  if (!std::any_of(allowed_ops.begin(), allowed_ops.end(),
                   [def_op](mlir::TypeID type_id) {
                     return def_op->getName().getTypeID() == type_id;
                   })) {
    return nullptr;
  }
  return get_prev_op<TargetOpType>(get_unique_operand_or_null(def_op),
                                   allowed_ops);
}

template <typename TargetOpType>
TargetOpType
get_succ_op(Value val, const std::initializer_list<mlir::TypeID> &allowed_ops) {
  if (val.use_empty())
    return nullptr;
  if (val.hasOneUse()) {
    auto user = *val.user_begin();
    if (auto target = dyn_cast<TargetOpType>(user)) {
      return target;
    }
    if (!std::any_of(allowed_ops.begin(), allowed_ops.end(),
                     [user](mlir::TypeID type_id) {
                       return user->getName().getTypeID() == type_id;
                     })) {
      return nullptr;
    }
    if (user->getNumResults() != 1) {
      llvm::errs()
          << "Warning: this api is designed for ONE-to-ONE op chain.\n";
      return nullptr;
    }
    return get_succ_op<TargetOpType>(user->getResult(0), allowed_ops);
  }
  return nullptr;
}

llvm::SmallVector<Value> split_value(Value in_value, int axis,
                                     std::vector<int64_t> &tile_lens,
                                     std::string suffix,
                                     PatternRewriter &rewriter);
llvm::SmallVector<Value> split_value(Value in_value, int axis, int tile_len,
                                     std::string suffix,
                                     PatternRewriter &rewriter);
Value clone_matmul(tpu::MatMulOp op, Value input, Value right, Value bias,
                   Value multi, PatternRewriter &rewriter, std::string suffix);
Value clone_splitK_matmul(tpu::MatMulOp matMulOp, Value input, Value weight,
                          Value bias, PatternRewriter &rewriter,
                          std::string suffix);
Value clone_reshape(tpu::ReshapeOp op, Value input,
                    std::vector<int64_t> &oshape, PatternRewriter &rewriter,
                    std::string suffix);
Value clone_common_op(Operation *op, Value input, PatternRewriter &rewriter,
                      std::string suffix);
Value clone_common_ops_between(Operation *beg_op, Operation *end_op,
                               Value input, PatternRewriter &rewriter,
                               std::string suffix, int max_depth = 10);
Value concat_values(llvm::SmallVector<Value> values, int axis,
                    PatternRewriter &rewriter);
Value reduce_add(Value input1, Value input2, PatternRewriter &rewriter);
Value apply_multipliers(Value input, Value multipliers, int scale, int rshift,
                        PatternRewriter &rewriter);

bool is_same_shape(const std::vector<int64_t> &shape1,
                   std::vector<int> exp_shape);

} // namespace tpu_dialect_tiling_primitives

class LargePadConvPattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  LargePadConvPattern(mlir::MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "LargePadConvPattern",
                                           benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::Conv2DOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

class PermuteReorderPattern : public OpRewriterPatternEx<tpu::PermuteOp> {
public:
  PermuteReorderPattern(mlir::MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteReorderPattern",
                                            benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::PermuteOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

struct PermutePadSwap : public OpRewriterPatternEx<tpu::PermuteOp> {
  PermutePadSwap(mlir::MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermutePadSwap",
                                            benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::PermuteOp op,
                      mlir::PatternRewriter &rewriter) const override;
};

Value createSplitQuantizedMLP(mlir::PatternRewriter &rewriter,
                              mlir::Operation *op, Value arg0);
Value createSplitQuantizedMLP2(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, Value arg0, int num_device);

struct RemoveReshape : public OpRewritePattern<tpu::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ReshapeOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace tpu
} // namespace tpu_mlir
