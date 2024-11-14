//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
using namespace tpu_mlir::top;

class LoopOpRewriteMaxTripCountPattern
    : public OpRewriterPatternEx<top::LoopOp> {
public:
  using OpRewriterPatternEx<top::LoopOp>::OpRewriterPatternEx;

  LoopOpRewriteMaxTripCountPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<LoopOp>(context,
                                    "LoopOpRewriteMaxTripCountPattern") {}

  LogicalResult matchAndRewriteImpl(top::LoopOp loopOp,
                                    PatternRewriter &rewriter) const override {
    Location loc = loopOp.getLoc();
    Operation *op = loopOp.getOperation();
    Value maxTripCountValue = op->getOperands()[0];

    /* when construct the ir at frontend, because Loop's body
       region is a graph, it have input node, will create inputOp
       at region, it will meet error when model_runner.py the top.mlir
       firstly, eliminate the inputOp */

    /* the frontend maybe generate the ir as below:
      %2 = "top.Squeeze"(%1) {axes = [0]} : (tensor<1xf32>) -> tensor<1xf32>
      loc(#loc3) %3 = "top.Weight"() : () -> tensor<1xf32> loc(#loc4) %4 =
      "top.Weight"() : () -> tensor<1xf32> loc(#loc5) %5 = "top.Weight"() : ()
      -> tensor<1xf32> loc(#loc6) %6:2 = "top.Loop"(%3, %2, %4, %5) ({
      ^bb0(%arg1: tensor<1xf32> loc(unknown), %arg2: tensor<1xf32> loc(unknown),
      %arg3: tensor<1xf32> loc(unknown), %arg4: tensor<1xf32> loc(unknown)): %9
      = "top.Input"(%arg3) : (tensor<1xf32>) -> tensor<1xf32> loc(#loc8) %10 =
      "top.Input"(%arg4) : (tensor<1xf32>) -> tensor<1xf32> loc(#loc9) %11 =
      "top.AddConst"(%10) {const_val = 2.000000e+00 : f64, do_relu = false,
      relu_limit = -1.000000e+00 : f64} : (tensor<1xf32>) -> tensor<1xf32>
      loc(#loc10) %12 = "top.Compare"(%11, %9) {mode = "Less"} : (tensor<1xf32>,
      tensor<1xf32>) -> tensor<1xf32> loc(#loc11) "top.Yield"(%12, %9, %11) :
      (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> () loc(#loc)
      }) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) ->
      (tensor<1xf32>, tensor<1xf32>) loc(#loc7)
    */
    Block &bodyBlock = loopOp.getBody().front();
    bodyBlock.walk<WalkOrder::PreOrder>([&](top::InputOp op) {
      op.replaceAllUsesWith(*(op.getODSOperands(0).begin()));
      rewriter.eraseOp(op);
    });
    // Match the following pattern:
    // ```
    // ubValue = WeightOp() {value = ...}
    // startValue = WeightOp() {value = ...}
    // Loop(max_trip_count, true, ..., ubValue, ..., startValue, ...)
    //   ^bb(max_trip_count, cond, ..., ubValue, ..., counterValue, ...):
    //     Note: stepValue also can be defined at above of Loop,
    //           and transfer to here by Loop's operand
    //     stepValue = WeightOp() {value = ...}
    //     newCounterValue = AddOp(counterValue, stepValue).
    //     cond_new = cond
    //     YieldOp (cond_new, ..., ubValue, ..., newCounterValue, ...)
    // ```
    bool matched;
    Value newMaxTripCountValue;
    std::tie(matched, newMaxTripCountValue) = matchOp(rewriter, loc, loopOp);
    if (!matched)
      return failure();

    // Rewrite
    op->replaceUsesOfWith(maxTripCountValue, newMaxTripCountValue);
    Region &loopBody = loopOp.getBody();
    Operation *loopBodyTerminator = loopBody.front().getTerminator();
    loopBodyTerminator->setOperand(0, loopBody.front().getArgument(1));
    return success();
  }

private:
  bool isDefinedByWeightOp(Value v) const {
    if (v.isa<BlockArgument>()) {
      auto index = v.cast<BlockArgument>().getArgNumber();
      auto vv = v.cast<BlockArgument>()
                    .getOwner()
                    ->getParentOp()
                    ->getOperands()[index];
      return isa<top::WeightOp>(vv.getDefiningOp());
    } else {
      Operation *definingOp = v.getDefiningOp();
      if (isa<top::WeightOp>(definingOp) &&
          v.getType()
              .cast<ShapedType>()
              .getElementType()
              .isa<IntegerType, Float32Type, Float16Type, BFloat16Type>())
        return true;
    }
    return false;
  }

  bool isInvariantBlockArg(Value v, Operation *returnOp) const {
    return v.isa<BlockArgument>() &&
           (v ==
            returnOp
                ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
  }

  bool isConstantOrInvariantBlockArg(Value v, Operation *returnOp) const {
    return ((v.isa<BlockArgument>() && isInvariantBlockArg(v, returnOp)) ||
            (!v.isa<BlockArgument>() && isDefinedByWeightOp(v)));
  }

  bool isUpdatedArgByValue(Value v, Value newV, Operation *returnOp) const {
    return v.isa<BlockArgument>() &&
           (newV ==
            returnOp
                ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
  }

  Value getFedValue(Value arg, Operation *op) const {
    return op->getOperands()[arg.cast<BlockArgument>().getArgNumber()];
  }

  int64_t getOneConstant(Value v) const {
    auto data = cast<top::WeightOp>(v.getDefiningOp()).read_as_float();
    return (int64_t)(data->data()[0]);
  }

  Value getSourceWeightV(Value v) const {
    /* frontend changed, replace
       addconst/compareconst with add/comare */
    if (isa<BlockArgument>(v)) {
      auto index = v.cast<BlockArgument>().getArgNumber();
      return v.cast<BlockArgument>()
          .getOwner()
          ->getParentOp()
          ->getOperands()[index];
    } else if (isa<top::WeightOp>(v.getDefiningOp())) {
      return v;
    } else {
      assert(0 && "fatal error, pls let us know ASAP.");
    }
  }

  std::pair<bool, Value> matchOp(PatternRewriter &rewriter, Location loc,
                                 top::LoopOp loopOp) const {
    Operation *op = loopOp.getOperation();
    Value maxTripCountValue = op->getOperands()[0];

    if (!isDefinedByWeightOp(maxTripCountValue))
      return std::make_pair(false, maxTripCountValue);

    Region &loopBody = loopOp.getBody();
    if (!loopBody.hasOneBlock())
      return std::make_pair(false, maxTripCountValue);

    Block &bodyBlock = loopBody.front();
    Operation *returnOp = bodyBlock.getTerminator();
    if (!isa<top::YieldOp>(returnOp))
      return std::make_pair(false, maxTripCountValue);

    Value breakCond;

    if (returnOp->getOperands()[0].isa<BlockArgument>()) {
      breakCond = returnOp->getOperands()[0];
    } else {
      llvm::TypeSwitch<Operation *>(returnOp->getOperands()[0].getDefiningOp())
          .Case<top::CompareConstOp, top::CompareOp>(
              [&](auto) { breakCond = returnOp->getOperands()[0]; })
          .Case<top::SqueezeOp>([&](auto) {
            breakCond = returnOp->getOperands()[0]
                            .getDefiningOp()
                            ->getPrevNode()
                            ->getResults()[0];
          })
          .Default(
              [](auto) { assert(0 && "fatal error, pls let us know ASAP."); });
    }

    if (breakCond.isa<BlockArgument>())
      return std::make_pair(false, maxTripCountValue);

    Operation *breakCondOp = breakCond.getDefiningOp();
    if (!isa<top::CompareConstOp, top::CompareOp>(breakCondOp))
      return std::make_pair(false, maxTripCountValue);

    Value newCounterValue = breakCondOp->getOperands()[0];
    Value ubValue;
    std::size_t compareConst = 1;
    /* the rhs(upbound) can be transfered
       by the loopOp operand */
    if (isa<top::CompareOp>(breakCondOp) &&
        !isDefinedByWeightOp(breakCondOp->getOperands()[1])) {
      ubValue = breakCondOp->getOperands()[1];
      compareConst = 0;
    }

    if (!newCounterValue.getType()
             .cast<ShapedType>()
             .getElementType()
             .isa<IntegerType, Float32Type, Float16Type, BFloat16Type>())
      return std::make_pair(false, maxTripCountValue);

    if (newCounterValue.isa<BlockArgument>() ||
        !isa<top::AddOp, top::AddConstOp>(newCounterValue.getDefiningOp()))
      return std::make_pair(false, maxTripCountValue);

    Operation *addOp = newCounterValue.getDefiningOp();
    Value counterValue = addOp->getOperands()[0];
    Value stepValue;
    std::size_t addConst = 1;
    /* steValue can be transfered to here by
       LoopOp's operand or aslo can be defined with WeightOp
       in current subgraph, it is defined by user */
    if (isa<top::AddOp>(newCounterValue.getDefiningOp()) &&
        !isDefinedByWeightOp(
            newCounterValue.getDefiningOp()->getOperands()[1])) {
      stepValue = newCounterValue.getDefiningOp()->getOperands()[1];
      addConst = 0;
    }

    // Counter is a block argument and updated at each iteration.
    if (!isUpdatedArgByValue(counterValue, newCounterValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Step must be a WeightOp inside the loop or an invariant argument.
    if (!addConst && !isConstantOrInvariantBlockArg(stepValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    Value lbValue = getFedValue(counterValue, loopOp);

    if (!compareConst && !isConstantOrInvariantBlockArg(ubValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    int ub_value = 0;
    if (!compareConst && isInvariantBlockArg(ubValue, returnOp))
      ubValue = getFedValue(ubValue, loopOp);
    else {
      if (isa<top::CompareConstOp>(breakCondOp)) {
        ub_value = (int)(cast<top::CompareConstOp>(breakCondOp)
                             .getConstVal()
                             .convertToDouble());
      } else {
        auto src_v = getSourceWeightV(breakCondOp->getOperands()[1]);
        ub_value = getOneConstant(src_v);
      }
    }

    int add_step_value = 0;
    if (!addConst && isInvariantBlockArg(stepValue, returnOp))
      stepValue = getFedValue(stepValue, loopOp);
    else {
      if (isa<top::AddConstOp>(addOp)) {
        add_step_value =
            (int)(cast<top::AddConstOp>(addOp).getConstVal().convertToDouble());
      } else {
        auto src_v = getSourceWeightV(addOp->getOperands()[1]);
        add_step_value = getOneConstant(src_v);
      }
    }

    // Case 1: the upper bound, lower bound and step are constants.
    if (isDefinedByWeightOp(lbValue) && compareConst && addConst) {
      int64_t lowerBound = getOneConstant(lbValue);
      int64_t upperBound = ub_value;
      int64_t step = add_step_value;
      if ((step <= 0) || (upperBound <= lowerBound))
        return std::make_pair(false, maxTripCountValue);
      int64_t derivedTripCount =
          std::ceil((1.0 * (upperBound - lowerBound)) / (1.0 * step));
      int64_t maxTripCount = getOneConstant(maxTripCountValue);

      if (maxTripCount <= derivedTripCount)
        return std::make_pair(false, maxTripCountValue);

      auto newMaxTripCount = std::make_shared<std::vector<float>>(1);
      newMaxTripCount->data()[0] = derivedTripCount;

      auto shape =
          maxTripCountValue.getType().template cast<ShapedType>().getShape();
      auto type = RankedTensorType::get(
          shape,
          maxTripCountValue.getType().cast<ShapedType>().getElementType());
      // create a new weightOp
      auto newValue = top::WeightOp::create(maxTripCountValue.getDefiningOp(),
                                            "_fp32", *newMaxTripCount, type);
      return std::make_pair(true, newValue);
    }

    // other case: Todo
    return std::make_pair(false, Value(nullptr));
  }
};

void LoopOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<LoopOpRewriteMaxTripCountPattern>(context);
}
