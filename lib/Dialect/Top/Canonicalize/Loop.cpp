//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
using namespace tpu_mlir::top;

class LoopOpRewriteMaxTripCountPattern : public OpRewritePattern<top::LoopOp> {
public:
  using OpRewritePattern<top::LoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      top::LoopOp loopOp, PatternRewriter &rewriter) const override {
    Location loc = loopOp.getLoc();
    Operation *op = loopOp.getOperation();
    Value maxTripCountValue = op->getOperands()[0];

    bool matched;
    Value newMaxTripCountValue;
    std::tie(matched, newMaxTripCountValue) =
        matchOp(rewriter, loc, loopOp);
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
  bool isBlockArgument(Value v) const {
    if (v.isa<BlockArgument>())
      return true;
    if (isa<top::InputOp>(v.getDefiningOp())
        && v.getDefiningOp()->getOperands()[0]
            .isa<BlockArgument>())
      return true;
    return false;
  }

  bool isDefinedByWeightOp(Value v) const {
    if (isBlockArgument(v))
      return false;
    Operation *definingOp = v.getDefiningOp();
    if (isa<top::WeightOp>(definingOp)
        && v.getType().cast<ShapedType>().
         getElementType().isa<IntegerType, Float32Type, Float16Type,
                               BFloat16Type>())
      return true;
    return false;
  }

  bool isInvariantBlockArg(Value v, Operation *returnOp) const {
    if (isBlockArgument(v)) {
      if (v.isa<BlockArgument>()) {
        return (v == returnOp
                   ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
      } else {
        return (v == returnOp
                   ->getOperands()[v.getDefiningOp()->getOperands()[0]
                      .cast<BlockArgument>().getArgNumber() - 1]);
      }
    } else {
      return false;
    }
  }

  bool isConstantOrInvariantBlockArg(Value v, Operation *returnOp) const {
    return ((isBlockArgument(v) && isInvariantBlockArg(v, returnOp)) ||
            (!isBlockArgument(v) && isDefinedByWeightOp(v)));
  }

  bool isUpdatedArgByValue(Value v, Value newV, Operation *returnOp) const {
    if (isBlockArgument(v)) {
      if (v.isa<BlockArgument>()) {
        return (newV ==
               returnOp
                   ->getOperands()[v.cast<BlockArgument>().getArgNumber() - 1]);
      } else {
        return (newV ==
               returnOp
                   ->getOperands()[v.getDefiningOp()->getOperands()[0]
                                   .cast<BlockArgument>().getArgNumber() - 1]);
      }
    } else {
      return false;
    }
  }

  Value getFedValue(Value arg, Operation *op) const {
    if (arg.isa<BlockArgument>())
      return op->getOperands()[arg.cast<BlockArgument>().getArgNumber()];
    else
      return op->getOperands()[arg.getDefiningOp()->getOperands()[0].
                               cast<BlockArgument>().getArgNumber()];
  }

  int64_t getOneConstant(Value v) const {
    auto data = cast<top::WeightOp>(v.getDefiningOp()).read_as_float();
    return (int64_t)(data->data()[0]);
  }

  std::pair<bool, Value> matchOp(
      PatternRewriter &rewriter, Location loc, top::LoopOp loopOp) const {
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

    /* the frontend maybe generate the ir as below:
      %10 = "top.Add"(%8, %9) {do_relu = false, relu_limit = -1.000000e+00 : f64} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32> loc(#loc9)
      %11 = "top.CompareConst"(%10) {const_val = 8.000000e+00 : f64, inversed = false, mode = "Less"} : (tensor<1xf32>) -> tensor<1xf32> loc(#loc10)
      %12 = "top.Squeeze"(%11) {axes = [0]} : (tensor<1xf32>) -> tensor<1xf32> loc(#loc11)
      "top.Yield"(%12, %10, %9, %8) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> () loc(#loc)
    */
    Value breakCond;
    if (isa<top::CompareConstOp, top::CompareOp>
            (returnOp->getOperands()[0].getDefiningOp()))
      breakCond = returnOp->getOperands()[0];
    else if (isa<top::SqueezeOp>(
          returnOp->getOperands()[0].getDefiningOp())) {
      breakCond = returnOp->getOperands()[0].getDefiningOp()
                    ->getPrevNode()->getResults()[0];
    } else {
      assert(0 &&
         "fatal error, pls let us know ASAP.");
    }

    if (isBlockArgument(breakCond))
      return std::make_pair(false, maxTripCountValue);

    Operation *breakCondOp = breakCond.getDefiningOp();
    if (!isa<top::CompareConstOp,
             top::CompareOp>(breakCondOp))
      return std::make_pair(false, maxTripCountValue);

    Value newCounterValue = breakCondOp->getOperands()[0];
    Value ubValue;
    std::size_t compare_const_flag = 1;
    if (isa<top::CompareOp>(breakCondOp)) {
      ubValue = breakCondOp->getOperands()[1];
      compare_const_flag = 0;
    }

    if (!newCounterValue.getType()
             .cast<ShapedType>()
             .getElementType()
             .isa<IntegerType, Float32Type,
                  Float16Type, BFloat16Type>())
      return std::make_pair(false, maxTripCountValue);

    if (isBlockArgument(newCounterValue)
        || !isa<top::AddOp, top::AddConstOp>(newCounterValue.getDefiningOp()))
      return std::make_pair(false, maxTripCountValue);

    Operation *addOp = nullptr;
    std::size_t add_const_flag = 1;
    if (isa<top::AddOp>(newCounterValue.getDefiningOp())) {
      addOp = cast<top::AddOp>(newCounterValue.getDefiningOp());
      add_const_flag = 0;
    } else if (isa<top::AddConstOp>(newCounterValue.getDefiningOp()))
      addOp = cast<top::AddConstOp>(newCounterValue.getDefiningOp());

    Value counterValue = addOp->getOperands()[0];
    Value stepValue;
    if (!add_const_flag) {
      stepValue = addOp->getOperands()[1];
    }

    // Counter is a block argument and updated at each iteration.
    if (!isUpdatedArgByValue(counterValue, newCounterValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    // Step must be a WeightOp inside the loop or an invariant argument.
    if (!add_const_flag
         && !isConstantOrInvariantBlockArg(stepValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    Value lbValue = getFedValue(counterValue, loopOp);

    if (!compare_const_flag
        && !isConstantOrInvariantBlockArg(ubValue, returnOp))
      return std::make_pair(false, maxTripCountValue);

    int ub_value = 0;
    if (!compare_const_flag
        && isInvariantBlockArg(ubValue, returnOp))
      ubValue = getFedValue(ubValue, loopOp);
    else {
      ub_value = (int)(cast<top::CompareConstOp>(breakCondOp)
                           .getConstVal().convertToDouble());
    }

    int add_step_value = 0;
    if (!add_const_flag
        && isInvariantBlockArg(stepValue, returnOp))
      stepValue = getFedValue(stepValue, loopOp);
    else
      add_step_value = (int)(cast<top::AddConstOp>(addOp).getConstVal().convertToDouble());

    // Case 1: the upper bound, lower bound and step are constants.
    if (isDefinedByWeightOp(lbValue)
        && compare_const_flag
        && add_const_flag) {
      int64_t lowerBound = getOneConstant(lbValue);
      int64_t upperBound = ub_value;
      int64_t step = add_step_value;
      if ((step <= 0) || (upperBound <= lowerBound))
        return std::make_pair(false, maxTripCountValue);
      int64_t derivedTripCount =
          ceil((1.0 * (upperBound - lowerBound)) / (1.0 * step));
      int64_t maxTripCount = getOneConstant(maxTripCountValue);

      if (maxTripCount <= derivedTripCount)
        return std::make_pair(false, maxTripCountValue);

      auto new_name = module::getName(maxTripCountValue.getDefiningOp()).str() + "_fp32";
      auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      auto shape = maxTripCountValue.getType().template cast<ShapedType>().getShape();
      auto type = RankedTensorType::get(shape,
                                      maxTripCountValue.getType().cast<ShapedType>().getElementType());
      auto newOp = rewriter.create<top::WeightOp>(name_loc,
                                                  type,
                                                  ValueRange{});

      TypeSwitch<Type>(maxTripCountValue.getType().cast<ShapedType>().getElementType())
        .Case<Float32Type, IntegerType, Float16Type,
              BFloat16Type>([&](Type) {
          std::vector<float> newValue(1, derivedTripCount);
          newOp.update(newValue, 1);
        })
        /*.Case<IntegerType>([&](Type){
          std::vector<float> newValue(1, derivedTripCount);
          newOp.update(newValue, 1);
        })*/;

      return std::make_pair(true, newOp.getResult());
    }

    //other case: Todo
    return std::make_pair(false, Value(nullptr));
  }
};

void LoopOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<LoopOpRewriteMaxTripCountPattern>(context);
}
