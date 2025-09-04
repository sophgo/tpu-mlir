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

void populateStructOptimizePatterns(RewritePatternSet *patterns,
                                    const std::vector<RewriterRule> &rules) {
  // Always-on fixed patterns (independent of external rules)
  // patterns->add<>(
  //     patterns->getContext(), 8);

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
