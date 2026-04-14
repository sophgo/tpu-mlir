//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
// This pass performs structural optimizations before shape-inference
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/RewriterConfigUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#define CONFIG_FILE_NAME                                                       \
  module::getName(module::getModuleOp()).str() + "_" +                         \
      module::getChipStr().str() + "_" + module::getModeStr() +                \
      ".struct_optimize.json"

#define ALT_CONFIG_FILE_NAME                                                   \
  module::getName(module::getModuleOp()).str() + "_" + "all" + "_" +           \
      module::getModeStr() + ".struct_optimize.json"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;
using namespace mlir;

namespace tpu_mlir {
namespace top {

// Helper function: Get batch_size from model inputs
// Before shape-infer, aggregate the 0th dimension from all input tensors.
// - Ignore inputs without rank or with dynamic 0th dim
// - If multiple known batch sizes exist and differ, warn and choose the maximum
// - If none are known, default to 1
inline int64_t getBatchSizeFromModelInput(Operation *op) {
  ModuleOp module_op = dyn_cast<ModuleOp>(op) ? dyn_cast<ModuleOp>(op)
                                              : op->getParentOfType<ModuleOp>();
  if (!module_op)
    return 1;

  auto main_func = module_op.lookupSymbol<FuncOp>("main");
  if (!main_func || main_func.getArguments().empty())
    return 1;

  int64_t chosenBatch = -1;
  bool conflict = false;

  for (auto arg : main_func.getArguments()) {
    auto shaped = arg.getType().dyn_cast<ShapedType>();
    if (!shaped || !shaped.hasRank() || shaped.getRank() == 0)
      continue;
    auto shape = shaped.getShape();
    int64_t b = (shape[0] == ShapedType::kDynamic) ? -1 : shape[0];
    if (b <= 0)
      continue;
    if (chosenBatch < 0) {
      chosenBatch = b;
    } else if (b != chosenBatch) {
      conflict = true;
      if (b > chosenBatch)
        chosenBatch = b;
    }
  }

  if (chosenBatch < 0)
    return 1;

  if (conflict) {
    llvm::errs() << "[WARN] Inconsistent batch sizes across inputs detected. "
                    "Using batch_size="
                 << chosenBatch << "\n";
  }

  return chosenBatch;
}

// Function: Remove Permute operation before LayerNorm
// Pattern:
// Permute(order=[1,0,2]) -> LayerNorm
// Optimization target:
// If the upstream of LayerNorm is Permute with order=[1,0,2], remove the
// Permute operation directly Matching conditions:
// 1. The input of LayerNormOp must be PermuteOp
// 2. The order of PermuteOp must be [1,0,2]
class ConvertRemovePermuteBeforeLayerNormPattern
    : public OpRewriterPatternEx4<LayerNormOp> {
public:
  ConvertRemovePermuteBeforeLayerNormPattern(
      mlir::MLIRContext *context, int benefit,
      const std::vector<RewriterRule> &rules)
      : OpRewriterPatternEx4<LayerNormOp>(
            context, "ConvertRemovePermuteBeforeLayerNormPattern", rules,
            benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::LayerNormOp op,
                      mlir::PatternRewriter &rewriter) const override {
    bool enable_pattern = false;
    if (!getPatternRules().empty()) {
      for (auto rule : getPatternRules()) {
        auto params = rule.params;
        if (getParam(params, "enable", false)) {
          enable_pattern = true;
          break;
        }
      }
    }
    if (!enable_pattern) {
      return failure();
    }

    auto input = op.getInput();
    auto permute_op = dyn_cast<top::PermuteOp>(input.getDefiningOp());
    if (!permute_op) {
      return failure();
    }
    auto order_array = module::getI64Array(permute_op.getOrderAttr());
    if (!order_array || order_array->size() != 3 || (*order_array)[0] != 1 ||
        (*order_array)[1] != 0 || (*order_array)[2] != 2) {
      return failure();
    }
    permute_op.getResult().replaceAllUsesWith(permute_op.getInput());
    rewriter.eraseOp(permute_op);
    return success();
  }
};

// Function: Remove Permute operation between Add and Gather
// Pattern:
// Add -> Permute -> Gather
// Optimization target:
// If the upstream of Gather is Permute, and the upstream of Permute is Add,
// remove the Permute operation directly Matching conditions:
// 1. The input of GatherOp must be PermuteOp
// 2. The input of PermuteOp must be AddOp
class ConvertRemovePermuteBetweenAddGatherPattern
    : public OpRewriterPatternEx4<GatherOp> {
public:
  ConvertRemovePermuteBetweenAddGatherPattern(
      mlir::MLIRContext *context, int benefit,
      const std::vector<RewriterRule> &rules)
      : OpRewriterPatternEx4<GatherOp>(
            context, "ConvertRemovePermuteBetweenAddGatherPattern", rules,
            benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(GatherOp op,
                      mlir::PatternRewriter &rewriter) const override {
    bool enable_pattern = false;
    if (!getPatternRules().empty()) {
      for (auto rule : getPatternRules()) {
        auto params = rule.params;
        if (getParam(params, "enable", false)) {
          enable_pattern = true;
          break;
        }
      }
    }
    if (!enable_pattern) {
      return failure();
    }
    auto input = op.getInput();
    auto permute_op = dyn_cast<top::PermuteOp>(input.getDefiningOp());
    if (!permute_op)
      return failure();
    auto add_op = dyn_cast<top::AddOp>(permute_op.getInput().getDefiningOp());
    if (!add_op)
      return failure();
    permute_op.getResult().replaceAllUsesWith(permute_op.getInput());
    rewriter.eraseOp(permute_op);
    return success();
  }
};

// Function: Fuse the chain Reshape->Unsqueeze->Permute->Squeeze->3*Gather into
// Reshape->3*Gather Optimization target: Simplify the above complex operation
// chain into Reshape(batch_size,reshape[0],reshape[2],reshape[3]) ->
// 3*Gather(axis=2) Matching conditions:
// 1. axes of SqueezeOp must be [3]
// 2. The input of SqueezeOp must be PermuteOp
// 3. The order of PermuteOp must be [3,1,2,0,4]
// 4. The input of PermuteOp must be UnsqueezeOp
// 5. axes of UnsqueezeOp must be [0]
// 6. The input of UnsqueezeOp must be ReshapeOp
// 7. The shape of ReshapeOp must have 4 dimensions, defined as (reshape[0],
// batch_size, reshape[2], reshape[3]), where batch_size is dynamic
// 8. The output of SqueezeOp must be reshape[2] GatherOps
// 9. axis of these GatherOps must be 0
// 10. batch_size is obtained from the 0th dimension of the model input

class ConvertFuseAttentionSlicePattern
    : public OpRewriterPatternEx4<SqueezeOp> {
public:
  ConvertFuseAttentionSlicePattern(mlir::MLIRContext *context, int benefit,
                                   const std::vector<RewriterRule> &rules)
      : OpRewriterPatternEx4<SqueezeOp>(
            context, "ConvertFuseAttentionSlicePattern", rules, benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::SqueezeOp op,
                      mlir::PatternRewriter &rewriter) const override {
    bool enable_pattern = false;
    if (!getPatternRules().empty()) {
      for (auto rule : getPatternRules()) {
        auto params = rule.params;
        if (getParam(params, "enable", false)) {
          enable_pattern = true;
          break;
        }
      }
    }
    if (!enable_pattern) {
      return failure();
    }
    auto squeeze_axes = module::getI64Array(op.getAxesAttr());
    if (!squeeze_axes || squeeze_axes->size() != 1 || (*squeeze_axes)[0] != 3) {
      return failure();
    }

    auto permute_op = dyn_cast<top::PermuteOp>(op.getOperand().getDefiningOp());
    if (!permute_op) {
      return failure();
    }

    auto permute_order = module::getI64Array(permute_op.getOrderAttr());
    if (!permute_order || permute_order->size() != 5 ||
        (*permute_order)[0] != 3 || (*permute_order)[1] != 1 ||
        (*permute_order)[2] != 2 || (*permute_order)[3] != 0 ||
        (*permute_order)[4] != 4) {
      return failure();
    }

    auto unsqueeze_op =
        dyn_cast<top::UnsqueezeOp>(permute_op.getInput().getDefiningOp());
    if (!unsqueeze_op) {
      return failure();
    }

    auto unsqueeze_axes = module::getI64Array(unsqueeze_op.getAxesAttr());
    if (!unsqueeze_axes || unsqueeze_axes->size() != 1 ||
        (*unsqueeze_axes)[0] != 0) {
      return failure();
    }

    auto reshape_op =
        dyn_cast<top::ReshapeOp>(unsqueeze_op.getInput().getDefiningOp());
    if (!reshape_op) {
      return failure();
    }

    auto reshape_shape = module::getI64Array(reshape_op.getShapeAttr());
    if (!reshape_shape || reshape_shape->size() != 4) {
      return failure();
    }

    int64_t batch_size = getBatchSizeFromModelInput(op);

    if ((*reshape_shape)[1] != batch_size) {
      return failure();
    }

    int64_t expected_gather_count = (*reshape_shape)[2];
    int gather_count = 0;
    for (auto user : op.getResult().getUsers()) {
      if (dyn_cast<top::GatherOp>(user)) {
        gather_count++;
      }
    }
    if (gather_count != expected_gather_count) {
      return failure();
    }

    for (auto user : op.getResult().getUsers()) {
      auto gather_op = dyn_cast<top::GatherOp>(user);
      if (gather_op && gather_op.getAxis() != 0) {
        return failure();
      }
    }

    for (auto user : op.getResult().getUsers()) {
      auto gather_op = dyn_cast<top::GatherOp>(user);
      if (gather_op) {
        gather_op->setAttr("axis", rewriter.getSI32IntegerAttr(2));
      }
    }

    int64_t new_shape_1 = (*reshape_shape)[0];
    int64_t new_shape_2 = (*reshape_shape)[2];
    int64_t new_shape_3 = (*reshape_shape)[3];

    auto original_input = reshape_op.getInput();
    auto input_shaped_type = original_input.getType().dyn_cast<ShapedType>();
    if (!input_shaped_type) {
      return failure();
    }

    auto new_reshape_type = RankedTensorType::get(
        {batch_size, new_shape_1, new_shape_2, new_shape_3},
        input_shaped_type.getElementType());

    rewriter.replaceOpWithNewOp<ReshapeOp>(
        op, new_reshape_type, original_input, Value(),
        rewriter.getI64ArrayAttr(
            {batch_size, new_shape_1, new_shape_2, new_shape_3}),
        -1);
    return success();
  }
};

// Function: Fuse and fix the chain MatMul->Permute->Reshape->MatMul->Reshape
// Pattern:
// MatMul -> Permute(order=[2,0,1,3]) ->
// Reshape(reshape[0]*batch_size,reshape[2]) -> MatMul ->
// Reshape(reshape[0],batch_size,reshape[2]) Optimization target:
// 1. Change the order of Permute from [2,0,1,3] to [0,2,1,3]
// 2. Change the shape of the first Reshape from
// [batch_size*reshape[0],reshape[2]] to [batch_size,reshape[0],reshape[2]]
// 3. Remove the second Reshape
// Matching conditions:
// 1. The shape of the latter ReshapeOp must be
//    [reshape[0], batch_size, reshape[2]], where batch_size is dynamic
// 2. Upstream of the latter ReshapeOp is a MatMulOp
// 3. The input of that MatMulOp is a ReshapeOp, whose shape is
//    [batch_size * reshape[0], reshape[2]]
// 4. The input of this ReshapeOp is a PermuteOp with order [2, 0, 1, 3]
// 5. (Relaxed) The input of the PermuteOp is no longer required to be MatMulOp
// 6. batch_size is obtained from the 0th dimension of the model input
class ConvertPermuteReshapeChainFixPattern
    : public OpRewriterPatternEx4<ReshapeOp> {
public:
  ConvertPermuteReshapeChainFixPattern(mlir::MLIRContext *context, int benefit,
                                       const std::vector<RewriterRule> &rules)
      : OpRewriterPatternEx4<ReshapeOp>(
            context, "ConvertPermuteReshapeChainFixPattern", rules, benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ReshapeOp op,
                      mlir::PatternRewriter &rewriter) const override {
    bool enable_pattern = false;
    if (!getPatternRules().empty()) {
      for (auto rule : getPatternRules()) {
        auto params = rule.params;
        if (getParam(params, "enable", false)) {
          enable_pattern = true;
          break;
        }
      }
    }
    if (!enable_pattern) {
      return failure();
    }
    auto shape = module::getI64Array(op.getShapeAttr());
    if (!shape || shape->size() != 3)
      return failure();
    int64_t last_dim0 = (*shape)[0];
    int64_t last_dim1 = (*shape)[1];
    int64_t last_dim2 = (*shape)[2];
    int64_t batch_size = getBatchSizeFromModelInput(op);
    if (last_dim1 != batch_size)
      return failure();
    auto matmul2 = dyn_cast<top::MatMulOp>(op.getInput().getDefiningOp());
    if (!matmul2)
      return failure();
    auto reshape1 =
        dyn_cast<top::ReshapeOp>(matmul2.getInput().getDefiningOp());
    if (!reshape1)
      return failure();
    auto shape1 = module::getI64Array(reshape1.getShapeAttr());
    if (!shape1 || shape1->size() != 2)
      return failure();
    int64_t r1_dim0 = (*shape1)[0];
    int64_t r1_dim1 = (*shape1)[1];
    if (r1_dim0 != batch_size * last_dim0 || r1_dim1 != last_dim2)
      return failure();
    auto permute =
        dyn_cast<top::PermuteOp>(reshape1.getInput().getDefiningOp());
    if (!permute)
      return failure();
    auto order = module::getI64Array(permute.getOrderAttr());
    if (!order || order->size() != 4 || (*order)[0] != 2 || (*order)[1] != 0 ||
        (*order)[2] != 1 || (*order)[3] != 3)
      return failure();
    permute->setAttr("order", rewriter.getI64ArrayAttr({0, 2, 1, 3}));
    reshape1->setAttr(
        "shape", rewriter.getI64ArrayAttr({batch_size, last_dim0, last_dim2}));
    op.getResult().replaceAllUsesWith(matmul2.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

// Function: Optimize the chain
// reshape(reshape[2],batch_size*reshape[1],reshape[3])->permute(order=[1,0,2])->reshape(batch_size,reshape[1],reshape[2],reshape[3])
// The chain becomes
// reshape(batch,reshape[2],reshape[1],reshape[3])->permute(order=[0,2,1,3])
// Pattern:
// Reshape(reshape[2],(batch_size*reshape[1]),reshape[3]) ->
// Permute(order=[1,0,2]) ->
// Reshape(batch_size,reshape[1],reshape[2],reshape[3]) Optimization target:
// Optimize the operation chain to Reshape(batch_size, reshape[2], reshape[1],
// reshape[3]) -> Permute(order=[0,2,1,3]) Matching conditions:
// 1. The shape of the latter ReshapeOp is
// (batch_size,reshape[1],reshape[2],reshape[3]), where batch_size is dynamic
// 2. The input of ReshapeOp must be PermuteOp
// 3. The order of PermuteOp must be [1,0,2]
// 4. The input of PermuteOp must be ReshapeOp
// 5. The shape of ReshapeOp must be
// (reshape[2],batch_size*reshape[1],reshape[3])
// 6. batch_size is obtained from the 0th dimension of the model input
class ConvertOptimizeReshapePermuteChainPattern
    : public OpRewriterPatternEx4<ReshapeOp> {
public:
  ConvertOptimizeReshapePermuteChainPattern(
      mlir::MLIRContext *context, int benefit,
      const std::vector<RewriterRule> &rules)
      : OpRewriterPatternEx4<ReshapeOp>(
            context, "ConvertOptimizeReshapePermuteChainPattern", rules,
            benefit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::ReshapeOp op,
                      mlir::PatternRewriter &rewriter) const override {
    bool enable_pattern = false;
    if (!getPatternRules().empty()) {
      for (auto rule : getPatternRules()) {
        auto params = rule.params;
        if (getParam(params, "enable", false)) {
          enable_pattern = true;
          break;
        }
      }
    }
    if (!enable_pattern) {
      return failure();
    }
    auto shape = module::getI64Array(op.getShapeAttr());
    if (!shape || shape->size() != 4)
      return failure();
    int64_t batch_size = getBatchSizeFromModelInput(op);
    if ((*shape)[0] != batch_size)
      return failure();
    int64_t dim1 = (*shape)[1];
    int64_t dim2 = (*shape)[2];
    int64_t dim3 = (*shape)[3];
    auto permute = dyn_cast<top::PermuteOp>(op.getInput().getDefiningOp());
    if (!permute)
      return failure();
    auto permute_order = module::getI64Array(permute.getOrderAttr());
    if (!permute_order || permute_order->size() != 3 ||
        (*permute_order)[0] != 1 || (*permute_order)[1] != 0 ||
        (*permute_order)[2] != 2)
      return failure();
    auto reshape = dyn_cast<top::ReshapeOp>(permute.getInput().getDefiningOp());
    if (!reshape)
      return failure();
    auto reshape_shape = module::getI64Array(reshape.getShapeAttr());
    if (!reshape_shape || reshape_shape->size() != 3)
      return failure();
    int64_t r2 = (*reshape_shape)[0];
    int64_t r1 = (*reshape_shape)[1];
    int64_t r3 = (*reshape_shape)[2];
    if (r1 != batch_size * dim1 || r2 != dim2 || r3 != dim3)
      return failure();
    auto op_name = module::getName(op.getOperation()).str();
    auto input_shaped_type =
        reshape.getInput().getType().dyn_cast<ShapedType>();
    if (!input_shaped_type)
      return failure();
    auto new_reshape_type = RankedTensorType::get(
        {batch_size, dim2, dim1, dim3}, input_shaped_type.getElementType());
    auto new_reshape = rewriter.create<ReshapeOp>(
        mlir::NameLoc::get(
            rewriter.getStringAttr(op_name + "_optimized_reshape")),
        new_reshape_type, reshape.getInput(),
        rewriter.getNamedAttr(
            "shape", rewriter.getI64ArrayAttr({batch_size, dim2, dim1, dim3})));
    rewriter.replaceOpWithNewOp<PermuteOp>(
        op, op.getType(), new_reshape.getOutput(),
        rewriter.getI64ArrayAttr({0, 2, 1, 3}));
    return success();
  }
};

// Pattern to fuse correlation computation chain into a single Correlation op
// Pattern scope:
// Start: Two operation outputs (any operation types)
// End: Final ScatterND operation
// Pattern content:
// 1. Initial operation: Mul(input1, input2) -> Reshape -> [Expand] ->
// ReduceMean -> Reshape -> ScatterND (offset=0 case)
// 2. Repeated operations: For i in [1, 2, ..., max_disp-1]:
//    Slice(input1, offset=[i]) -> Slice(input2, ends=[-i]) ->
//    Mul -> Reshape -> [Expand] -> ReduceMean -> Reshape -> ScatterND
// Optimization goal:
// Replace entire correlation computation with Correlation(input1, input2) ->
// Unsqueeze -> subsequent operations
class ConvertFuseCorrelationPattern
    : public OpRewritePattern<top::ScatterNDOp> {
public:
  ConvertFuseCorrelationPattern(mlir::MLIRContext *context,
                                PatternBenefit benefit)
      : OpRewritePattern<top::ScatterNDOp>(context, benefit) {}

  LogicalResult matchAndRewrite(top::ScatterNDOp op,
                                PatternRewriter &rewriter) const override {
    // Step 1: Check if this ScatterND has users (output is used by other
    // operations)
    if (op.getResult().getUsers().empty()) {
      return failure();
    }

    // Step 2: Check if Correlation operation already exists (avoid duplicate
    // processing)
    auto block = op->getBlock();
    for (auto &block_op : *block) {
      if (isa<top::CorrelationOp>(&block_op)) {
        return failure();
      }
    }

    // Step 3: Trace back the correlation chain to find the initial two source
    // operations Trace from final ScatterND: ScatterND <- Reshape <- [Expand]
    // <- ReduceMean <- Reshape <- Mul <- Slice/any_op
    auto updates = op.getUpdates();

    auto reshape2 = dyn_cast<top::ReshapeOp>(updates.getDefiningOp());
    if (!reshape2) {
      return failure();
    }

    // Check for optional Expand operation after Reshape
    Value reduce_input = reshape2.getInput();
    auto expand = dyn_cast<top::ExpandOp>(reduce_input.getDefiningOp());
    if (expand) {
      reduce_input = expand.getInput();
    }

    auto reduce = dyn_cast<top::ReduceOp>(reduce_input.getDefiningOp());
    if (!reduce || reduce.getMode() != "ReduceMean") {
      return failure();
    }

    Value mul_input;
    auto reshape1 = dyn_cast<top::ReshapeOp>(reduce.getInput().getDefiningOp());
    if (reshape1) {
      mul_input = reshape1.getInput();
    } else {
      mul_input = reduce.getInput();
    }

    auto mul = dyn_cast<top::MulOp>(mul_input.getDefiningOp());
    if (!mul) {
      return failure();
    }

    // Step 4: Determine two source operations from Mul inputs
    auto mul_input1 = mul.getInputs()[0];
    auto mul_input2 = mul.getInputs()[1];

    Value left_input, right_input;

    // Case 1: Direct Mul of two operation outputs (offset=0 case)
    if (mul_input1.getDefiningOp() && mul_input2.getDefiningOp()) {
      // Check if both are not Slice operations (direct use of source operation
      // outputs)
      auto slice1 = dyn_cast<top::SliceOp>(mul_input1.getDefiningOp());
      auto slice2 = dyn_cast<top::SliceOp>(mul_input2.getDefiningOp());

      if (!slice1 && !slice2) {
        // This is the initial Mul operation for offset=0
        left_input = mul_input1;
        right_input = mul_input2;
      } else if (slice1 && slice2) {
        // Case 2: Mul of two Slice outputs (offset>0 case)
        // Trace back from Slice to source operations
        left_input = slice1.getInput();
        right_input = slice2.getInput();

        // Verify these inputs have defining operations
        if (!left_input.getDefiningOp() || !right_input.getDefiningOp()) {
          return failure();
        }
      } else {
        return failure();
      }
    } else {
      return failure();
    }

    // Step 5: Scan entire block to collect all related correlation operations
    std::set<int64_t> all_offsets;
    std::vector<top::ScatterNDOp> all_scatter_ops;

    // Traverse all ScatterND operations in block to find operations belonging
    // to same correlation pattern
    for (auto &block_op : *block) {
      if (auto scatter_op = dyn_cast<top::ScatterNDOp>(&block_op)) {
        // Trace back to verify if this ScatterND belongs to the same
        // correlation pattern
        auto sc_updates = scatter_op.getUpdates();
        auto sc_reshape2 = dyn_cast<top::ReshapeOp>(sc_updates.getDefiningOp());
        if (!sc_reshape2)
          continue;

        // Check for optional Expand operation after Reshape
        Value sc_reduce_input = sc_reshape2.getInput();
        auto sc_expand =
            dyn_cast<top::ExpandOp>(sc_reduce_input.getDefiningOp());
        if (sc_expand) {
          sc_reduce_input = sc_expand.getInput();
        }

        auto sc_reduce =
            dyn_cast<top::ReduceOp>(sc_reduce_input.getDefiningOp());
        if (!sc_reduce || sc_reduce.getMode() != "ReduceMean")
          continue;

        Value sc_mul_input;
        auto sc_reshape1 =
            dyn_cast<top::ReshapeOp>(sc_reduce.getInput().getDefiningOp());
        if (sc_reshape1) {
          sc_mul_input = sc_reshape1.getInput();
        } else {
          sc_mul_input = sc_reduce.getInput();
        }

        auto sc_mul = dyn_cast<top::MulOp>(sc_mul_input.getDefiningOp());
        if (!sc_mul)
          continue;

        // Check if this Mul's inputs are related to our left_input and
        // right_input
        auto sc_input1 = sc_mul.getInputs()[0];
        auto sc_input2 = sc_mul.getInputs()[1];

        bool belongs_to_pattern = false;

        // Check for offset=0 case (direct use of source operation outputs)
        if ((sc_input1 == left_input && sc_input2 == right_input) ||
            (sc_input1 == right_input && sc_input2 == left_input)) {
          belongs_to_pattern = true;
          all_offsets.insert(0);
        } else {
          // Check for offset>0 case (use of Slice outputs)
          auto sc_slice1 = dyn_cast<top::SliceOp>(sc_input1.getDefiningOp());
          auto sc_slice2 = dyn_cast<top::SliceOp>(sc_input2.getDefiningOp());

          if (sc_slice1 && sc_slice2) {
            if ((sc_slice1.getInput() == left_input &&
                 sc_slice2.getInput() == right_input) ||
                (sc_slice1.getInput() == right_input &&
                 sc_slice2.getInput() == left_input)) {
              belongs_to_pattern = true;

              // Extract offset information
              auto left_slice =
                  (sc_slice1.getInput() == left_input) ? sc_slice1 : sc_slice2;
              auto offset_array =
                  module::getI64Array(left_slice.getOffsetAttr());
              if (offset_array && offset_array->size() == 1) {
                all_offsets.insert((*offset_array)[0]);
              }
            }
          }
        }

        if (belongs_to_pattern) {
          all_scatter_ops.push_back(scatter_op);
        }
      }
    }

    // Calculate max_disp
    int64_t max_disp = all_offsets.size();
    if (max_disp == 0) {
      return failure();
    }

    // Infer num_groups from reshape2
    int64_t num_groups = 0;
    auto shape_array = module::getI64Array(reshape2.getShapeAttr());
    if (!shape_array || shape_array->size() < 3) {
      return failure();
    }
    num_groups = (*shape_array)[1]; // Second dimension is num_groups
    if (num_groups <= 0) {
      return failure();
    }

    // Step 6: Find the initial Weight tensor to determine output dimensions
    // Trace back through ScatterND chain to find the first ScatterND's data
    // input
    int64_t target_output_dims = 0;
    std::vector<int64_t> weight_shape;
    Operation *current_scatter = op;
    while (current_scatter) {
      auto scatter_op = dyn_cast<top::ScatterNDOp>(current_scatter);
      if (!scatter_op)
        break;

      auto data_input = scatter_op.getInputData();
      auto data_def_op = data_input.getDefiningOp();

      // Check if data input is a Weight (initial tensor) or another ScatterND
      if (auto weight_op = dyn_cast<top::WeightOp>(data_def_op)) {
        // Found the initial Weight tensor, get its dimensions
        if (auto shaped_type = data_input.getType().dyn_cast<ShapedType>()) {
          if (shaped_type.hasRank()) {
            target_output_dims = shaped_type.getRank();
            for (auto dim : shaped_type.getShape()) {
              weight_shape.push_back(dim);
            }
          }
        }
        break;
      } else if (auto prev_scatter = dyn_cast<top::ScatterNDOp>(data_def_op)) {
        // Continue tracing back
        current_scatter = prev_scatter;
      } else {
        break;
      }
    }

    // If we couldn't determine from Weight, fall back to input dimensions
    if (target_output_dims == 0) {
      if (auto shaped_type = left_input.getType().dyn_cast<ShapedType>()) {
        if (shaped_type.hasRank()) {
          target_output_dims = shaped_type.getRank();
        }
      }
    }

    // Step 7: Get input shape to determine if we need 5D->4D reshape
    // Since input types may be unranked (tensor<*xf32>), we infer dimensions
    // from weight_shape
    auto left_shaped_type = left_input.getType().dyn_cast<ShapedType>();
    int64_t input_dims = 0;
    std::vector<int64_t> input_shape_vec;

    if (left_shaped_type && left_shaped_type.hasRank()) {
      auto input_shape = left_shaped_type.getShape();
      input_dims = input_shape.size();
      for (auto dim : input_shape) {
        input_shape_vec.push_back(dim);
      }
    } else {
      // Input is unranked, infer from weight_shape
      // weight_shape is [B, num_groups, max_disp, H, W]
      // input_shape should be [B, num_groups, C, H, W] where C is inferred
      if (weight_shape.size() == 5) {
        input_dims = 5;
        // Trace back to find a Reshape op with 5D shape attribute
        // Use generic traversal - follow first input of any op until we find
        // Reshape
        Value trace_val = left_input;
        bool found_shape = false;
        for (int i = 0; i < 20 && !found_shape; ++i) {
          auto def_op = trace_val.getDefiningOp();
          if (!def_op)
            break;

          if (auto reshape_op = dyn_cast<top::ReshapeOp>(def_op)) {
            auto shape_attr = module::getI64Array(reshape_op.getShapeAttr());
            if (shape_attr && shape_attr->size() == 5) {
              for (auto dim : *shape_attr) {
                input_shape_vec.push_back(dim);
              }
              found_shape = true;
              break;
            }
            trace_val = reshape_op.getInput();
          } else if (def_op->getNumOperands() > 0) {
            // Generic: follow first operand of any op
            trace_val = def_op->getOperand(0);
          } else {
            break;
          }
        }

        if (!found_shape) {
          return failure();
        }
      } else {
        return failure();
      }
    }

    // Check if we have valid input shape
    if (input_shape_vec.empty() || input_dims == 0) {
      llvm::errs() << "  FAIL: Could not determine input shape\n";
      return failure();
    }

    // Infer element type from input type (use f32 as default for unranked
    // tensors)
    Type element_type = rewriter.getF32Type();
    if (left_shaped_type) {
      element_type = left_shaped_type.getElementType();
    }

    auto max_disp_attr = rewriter.getI64IntegerAttr(max_disp);
    auto num_groups_attr = rewriter.getI64IntegerAttr(num_groups);

    Value corr_left_input = left_input;
    Value corr_right_input = right_input;
    int64_t batch_size = 1;

    // Step 8: Handle 5D input case - reshape to 4D before Correlation
    // 5D input: [B, num_groups, C, H, W] -> 4D: [B, num_groups*C, H, W]
    if (input_dims == 5 && input_shape_vec.size() == 5) {
      batch_size = input_shape_vec[0];
      int64_t ng = input_shape_vec[1]; // num_groups
      int64_t c = input_shape_vec[2];  // C
      int64_t h = input_shape_vec[3];  // H
      int64_t w = input_shape_vec[4];  // W

      // Create 4D shape: [B, num_groups*C, H, W]
      std::vector<int64_t> reshape_4d_shape = {batch_size, ng * c, h, w};
      auto reshape_4d_type =
          RankedTensorType::get(reshape_4d_shape, element_type);

      // Create Reshape for left input: 5D -> 4D
      auto left_reshape_op = rewriter.create<top::ReshapeOp>(
          mlir::NameLoc::get(
              rewriter.getStringAttr("corr_left_reshape_5d_to_4d")),
          reshape_4d_type, left_input);
      corr_left_input = left_reshape_op.getResult();

      // Create Reshape for right input: 5D -> 4D
      auto right_reshape_op = rewriter.create<top::ReshapeOp>(
          mlir::NameLoc::get(
              rewriter.getStringAttr("corr_right_reshape_5d_to_4d")),
          reshape_4d_type, right_input);
      corr_right_input = right_reshape_op.getResult();
    }

    // Step 9: Create Correlation operation (with 4D inputs)
    auto correlation_type = UnrankedTensorType::get(element_type);

    auto correlation_op = rewriter.create<top::CorrelationOp>(
        mlir::NameLoc::get(rewriter.getStringAttr("fused_correlation")),
        correlation_type, ValueRange{corr_left_input, corr_right_input},
        max_disp_attr, num_groups_attr);

    Value final_output = correlation_op.getResult();

    // Step 10: Handle output dimension mismatch
    // Correlation output is 4D: [num_groups, max_disp, H, W]
    // Target may be 5D: [B, num_groups, max_disp, H, W]
    if (target_output_dims == 5 && weight_shape.size() == 5) {
      if (input_dims == 5) {
        // 5D input case: reshape Correlation 4D output back to 5D
        auto reshape_5d_type =
            RankedTensorType::get(weight_shape, element_type);
        auto output_reshape_op = rewriter.create<top::ReshapeOp>(
            mlir::NameLoc::get(rewriter.getStringAttr("corr_reshape_4d_to_5d")),
            reshape_5d_type, final_output);
        final_output = output_reshape_op.getResult();
      } else if (input_dims == 4) {
        // 4D input case: Unsqueeze to add batch dimension
        // Correlation output 4D -> Unsqueeze axes=[0] -> 5D
        auto unsqueeze_type = UnrankedTensorType::get(element_type);
        auto axes_attr = rewriter.getI64ArrayAttr({0});
        auto unsqueeze_op = rewriter.create<top::UnsqueezeOp>(
            mlir::NameLoc::get(rewriter.getStringAttr("corr_unsqueeze_batch")),
            unsqueeze_type, final_output, axes_attr);
        final_output = unsqueeze_op.getResult();
      }
    }

    // Step 11: Replace the final ScatterND with final output
    rewriter.replaceOp(op, final_output);
    return success();
  }
};

// Pattern to fuse ConcatVolume computation chain into a single ConcatVolume op
// ConcatVolume is used in stereo matching to build concatenation cost volume
//
// The ConcatVolume pattern in MLIR looks like a chain of ScatterND operations:
// Pattern structure (from actual MLIR):
// 1. Two Conv outputs: left_feature and right_feature
// 2. Initial Weight tensor with shape [1, 2*C, max_disp, H, W] (zeros)
// 3. For disparity d=0:
//    - left_feature -> Expand -> Reshape[1,C,1,H,W] -> ScatterND
//    - right_feature -> Expand -> Reshape[1,C,1,H,W] -> ScatterND
// 4. For disparity d>0:
//    - left_feature -> Expand -> Reshape[1,C,1,H,W] -> ScatterND (same as d=0)
//    - right_feature -> Slice(ends=[-d]) -> Expand -> Reshape[1,C,1,H,W-d] ->
//    ScatterND
//
// Key observations:
// - Slice uses axes=[3] (width dimension) with ends=[-d] for d>0
// - left_feature is never sliced, right_feature is sliced for d>0
// - Each disparity level has 2 ScatterND operations (one for left, one for
// right)
class ConvertFuseConcatVolumePattern
    : public OpRewritePattern<top::ScatterNDOp> {
public:
  ConvertFuseConcatVolumePattern(mlir::MLIRContext *context,
                                 PatternBenefit benefit)
      : OpRewritePattern<top::ScatterNDOp>(context, benefit) {}

  LogicalResult matchAndRewrite(top::ScatterNDOp op,
                                PatternRewriter &rewriter) const override {
    // Step 1: Check if this is the LAST ScatterND in the chain
    // (not used by another ScatterND)
    bool is_final_scatter = true;
    for (auto user : op.getResult().getUsers()) {
      if (dyn_cast<top::ScatterNDOp>(user)) {
        is_final_scatter = false;
        break;
      }
    }

    if (!is_final_scatter) {
      return failure();
    }

    auto block = op->getBlock();

    // Step 2: Trace back through the ScatterND chain to collect all operations
    std::vector<top::ScatterNDOp> scatter_chain;
    llvm::DenseSet<Value> source_inputs;
    int64_t max_slice_offset = 0;
    Value sliced_input; // The input that gets sliced (right_feature)

    // llvm::errs() << "=== ConcatVolume Pattern Matching Debug ===\n";

    Operation *current_op = op;
    while (current_op) {
      auto scatter_op = dyn_cast<top::ScatterNDOp>(current_op);
      if (!scatter_op)
        break;

      scatter_chain.push_back(scatter_op);

      // Analyze the updates input:
      // Pattern 1: Reshape <- Expand <- [Slice] <- source (with Expand)
      // Pattern 2: Reshape <- [Slice] <- source (without Expand, for
      // ConcatVolume)
      // Pattern 3: Reshape <- Slice(width) <- Slice(channel) <- source
      // (ConcatVolume with channel slice first)
      // Pattern 4: Reshape <- Slice(channel) <- source (no width slice, d=0)
      auto updates = scatter_op.getUpdates();
      auto reshape = dyn_cast<top::ReshapeOp>(updates.getDefiningOp());
      if (reshape) {
        auto reshape_shape = module::getI64Array(reshape.getShapeAttr());
        // Check if this is a 5D reshape with shape [1, C, 1, H, W]
        if (reshape_shape && reshape_shape->size() == 5 &&
            (*reshape_shape)[2] == 1) {
          Value reshape_input = reshape.getInput();
          Value source;

          // Try Pattern 1: Reshape <- Expand <- [Slice] <- source
          auto expand = dyn_cast<top::ExpandOp>(reshape_input.getDefiningOp());
          if (expand) {
            source = expand.getInput();
          } else {
            // Pattern 2/3/4: Reshape <- [Slice] <- source (no Expand)
            source = reshape_input;
          }

          // Check if source is a Slice operation
          auto slice = dyn_cast<top::SliceOp>(source.getDefiningOp());
          if (slice) {
            // Check if this is a width slice (ends=[..., -d]) or channel slice
            auto ends_array = module::getI64Array(slice.getEndsAttr());
            bool is_width_slice = false;
            // llvm::errs() << "  Found Slice, ends=[";
            // if (ends_array) {
            //   for (size_t i = 0; i < ends_array->size(); i++) {
            //     llvm::errs() << (*ends_array)[i];
            //     if (i < ends_array->size() - 1)
            //       llvm::errs() << ",";
            //   }
            // }
            // llvm::errs() << "]\n";

            if (ends_array && !ends_array->empty()) {
              // Check the last element for width slice (ends=[..., -d])
              int64_t end_val = ends_array->back();
              if (end_val < 0) {
                // This is a width slice with negative end
                is_width_slice = true;
                int64_t offset = -end_val;
                // llvm::errs() << "  -> Width slice detected, offset=" <<
                // offset
                //              << "\n";
                if (offset > max_slice_offset) {
                  max_slice_offset = offset;
                }
                // The sliced input is the input to this width slice
                // (which is the channel-sliced tensor)
                sliced_input = slice.getInput();
              }
            }

            if (is_width_slice) {
              // Pattern 3: Reshape <- Slice(width) <- Slice(channel) <- source
              // Move source to the input of the width slice (channel-sliced
              // tensor)
              source = slice.getInput();
              // llvm::errs() << "  -> Source updated to width slice input\n";
              // source is now the channel-sliced tensor, add it to
              // source_inputs This is the "right" feature that gets
              // width-sliced
            } else {
              // Pattern 4: Reshape <- Slice(channel) <- source (no width slice)
              // This is for d=0 case, source is the channel-sliced tensor
              // Don't move source further, keep it as the channel-sliced result
              // Actually, we want the channel-sliced tensor as source
              // source is already pointing to the channel slice result
              // llvm::errs() << "  -> Channel slice (d=0), source unchanged\n";
            }
          } else {
            // llvm::errs() << "  No Slice found for reshape input\n";
          }
          // llvm::errs() << "  Adding source to source_inputs\n";
          source_inputs.insert(source);
        }
      }

      // Move to the input ScatterND (the data input)
      auto data_input = scatter_op.getInputData();
      current_op = data_input.getDefiningOp();
    }

    // Step 3: Validate the pattern
    // llvm::errs() << "=== Validation ===\n";
    // llvm::errs() << "  scatter_chain.size() = " << scatter_chain.size() <<
    // "\n"; llvm::errs() << "  source_inputs.size() = " << source_inputs.size()
    // << "\n"; llvm::errs() << "  max_slice_offset = " << max_slice_offset <<
    // "\n";

    // We need exactly 2 source inputs (left and right features)
    if (source_inputs.size() != 2) {
      // llvm::errs() << "  FAILED: source_inputs.size() != 2\n";
      return failure();
    }

    // The chain should have at least 4 ScatterND operations
    if (scatter_chain.size() < 4) {
      return failure();
    }

    // Calculate max_disp from the maximum slice offset
    // max_slice_offset is the largest d value, so max_disp = max_slice_offset +
    // 1
    int64_t max_disp = max_slice_offset + 1;

    // Validate: number of ScatterND should be approximately 2 * max_disp
    if (scatter_chain.size() < (size_t)(max_disp * 2)) {
      return failure();
    }

    // Step 4: Check if ConcatVolume already exists for these inputs
    std::vector<Value> input_vec(source_inputs.begin(), source_inputs.end());
    for (auto &block_op : *block) {
      if (auto concat_vol_op = dyn_cast<top::ConcatVolumeOp>(&block_op)) {
        auto inputs = concat_vol_op.getInputs();
        if (inputs.size() == 2) {
          bool match1 =
              (inputs[0] == input_vec[0] && inputs[1] == input_vec[1]);
          bool match2 =
              (inputs[0] == input_vec[1] && inputs[1] == input_vec[0]);
          if (match1 || match2) {
            return failure(); // Already processed
          }
        }
      }
    }

    // Step 5: Determine which input is left and which is right
    // The right input is the one that gets sliced (sliced_input)
    // The left input is the other one
    Value left_input, right_input;

    if (sliced_input) {
      right_input = sliced_input;
      for (auto &src : source_inputs) {
        if (src != right_input) {
          left_input = src;
          break;
        }
      }
    } else {
      // Fallback: scan block for Slice operations
      for (auto &block_op : *block) {
        if (auto slice_op = dyn_cast<top::SliceOp>(&block_op)) {
          Value slice_source = slice_op.getInput();
          if (source_inputs.count(slice_source)) {
            right_input = slice_source;
            break;
          }
        }
      }
      for (auto &src : source_inputs) {
        if (src != right_input) {
          left_input = src;
          break;
        }
      }
    }

    if (!left_input || !right_input) {
      // If we still can't determine, use the order from the set
      left_input = input_vec[0];
      right_input = input_vec[1];
    }

    // Step 6: Create ConcatVolume operation
    auto max_disp_attr = rewriter.getI64IntegerAttr(max_disp);

    // Infer element type from input type
    Type element_type = rewriter.getF32Type();
    if (auto shaped_type = left_input.getType().dyn_cast<ShapedType>()) {
      element_type = shaped_type.getElementType();
    }

    auto concat_volume_type = UnrankedTensorType::get(element_type);

    auto concat_volume_op = rewriter.create<top::ConcatVolumeOp>(
        mlir::NameLoc::get(rewriter.getStringAttr("fused_concat_volume")),
        concat_volume_type, ValueRange{left_input, right_input}, max_disp_attr);

    // Step 7: Replace final ScatterND operation
    rewriter.replaceOp(op, concat_volume_op.getResult());

    return success();
  }
};

void populateStructOptimizePatterns(RewritePatternSet *patterns,
                                    const std::vector<RewriterRule> &rules) {
  // Always-on fixed patterns (independent of external rules)
  patterns->add<ConvertFuseCorrelationPattern>(patterns->getContext(), 8);
  patterns->add<ConvertFuseConcatVolumePattern>(patterns->getContext(), 8);

  // Rule-driven patterns (enabled only when rules are provided)
  if (!rules.empty()) {
    patterns->add<ConvertFuseAttentionSlicePattern,
                  ConvertPermuteReshapeChainFixPattern,
                  ConvertRemovePermuteBeforeLayerNormPattern,
                  ConvertRemovePermuteBetweenAddGatherPattern,
                  ConvertOptimizeReshapePermuteChainPattern>(
        patterns->getContext(), 8, rules);
  }
}

// Main Pass
class StructOptimizePass
    : public PassWrapper<StructOptimizePass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "struct-optimize"; }

  StringRef getDescription() const override {
    return "CLIP text branch struct optimization";
  }

  void runOnOperation() override {
    auto mOp = getOperation();

    int64_t batch_size = getBatchSizeFromModelInput(mOp);
    PASS_LOG_DEBUG_BLOCK(
        { rso << "[DEBUG] detected batch_size = " << batch_size; });

    std::string configPath = CONFIG_FILE_NAME;

    if (module::getChipStr().str() == "ALL") {
      std::string altConfigPath = ALT_CONFIG_FILE_NAME;
      auto altRules = loadRewriteConfig(altConfigPath);
      if (!altRules.empty()) {
        configPath = altConfigPath;
      }
    }

    if (getenv("TOP_DIALECT_REWRITER_CONFIG")) {
      configPath = std::string(getenv("TOP_DIALECT_REWRITER_CONFIG"));
    }

    auto rules = loadRewriteConfig(configPath);

    PASS_LOG_DEBUG_BLOCK({ dumpRewriterRules(rules, llvm::outs(), true); });

    RewritePatternSet patterns(&getContext());
    populateStructOptimizePatterns(&patterns, rules);
    (void)applyPatternsAndFoldGreedily(mOp, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStructOptimizePass() {
  return std::make_unique<StructOptimizePass>();
}

} // namespace top
} // namespace tpu_mlir
