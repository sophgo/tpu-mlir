//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/DoExtraOpt.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "cv18xx_patterns"
namespace tpu_mlir {
namespace cv18xx {
  LogicalResult FuseLeakReluPattern::matchAndRewrite(tpu::LeakyReluOp leakyReluOp,
                                PatternRewriter &rewriter) const  {
    assert(leakyReluOp);
    auto preOp = leakyReluOp.getInput().getDefiningOp();
    if (auto convOp = dyn_cast<tpu::Conv2DOp>(preOp)) {
      if (!module::isUniformQuantized(convOp.getOutput()) ||
          !module::isUniformQuantized(leakyReluOp.getOutput())) {
        return failure();
      }
      convOp->setAttr("do_leaky_relu", rewriter.getBoolAttr(true));
      convOp->setAttr("neg_slope", leakyReluOp.getAlphaAttr());
      if (leakyReluOp.getRshift().has_value())
        convOp->setAttr("rshift_pos", leakyReluOp.getRshiftAttr());
      if (leakyReluOp.getMultiplier().has_value())
        convOp->setAttr("multiplier_pos", leakyReluOp.getMultiplierAttr());
      if (leakyReluOp.getRshiftNeg().has_value())
        convOp->setAttr("rshift_neg", leakyReluOp.getRshiftNegAttr());
      if (leakyReluOp.getMultiplierNeg().has_value())
        convOp->setAttr("multiplier_neg", leakyReluOp.getMultiplierNegAttr());
      convOp->setLoc(leakyReluOp.getLoc());
      // remove the relu Op
      rewriter.replaceOp(leakyReluOp, {leakyReluOp.getInput()});
      return success();
    }

    return failure();
  }
} // namespace cv18xx
} // namespace tpu_mlir
