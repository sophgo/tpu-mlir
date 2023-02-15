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

LogicalResult MoveConvStrideToEltwiseOpPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {

  if (!op->hasTrait<trait::SupportEarlyStride>()) {
    return failure();
  }
  Operation *nextOp = nullptr;
  int strideH = 0;
  int strideW = 0;
  for (auto &use : op->getResult(0).getUses()) {
    nextOp = use.getOwner();
    if (auto convOp = dyn_cast<tpu::Conv2DOp>(nextOp)) {
      auto attrs = convOp.parseParam();
      int kh = attrs.kh;
      int kw = attrs.kw;
      int sh = attrs.sh;
      int sw = attrs.sw;
      if (kw == 0) {
        return failure();
      }
      if (strideH == 0 || strideW == 0) {
        strideH = sh;
        strideW = sw;
      }
      if (strideH != sh || strideW != sw) {
        LLVM_DEBUG(llvm::errs()
                   << "stride of all successor conv2d should be same\n");
        return failure();
      }
      if (sh == 1 || sw == 1) {
        return failure();
      }
      if (kh != 1 || kw != 1) {
        return failure();
      }
    } else {
      // if one of uses is not 1x1 conv,
      // we cannot do early stride.
      return failure();
    }
  }

  auto shape = module::getShape(op->getResult(0));
  if (shape[2] % strideH != 0 || shape[3] % strideW != 0) {
    // padding case, stop
    return failure();
  }

  for (auto &use : op->getResult(0).getUses()) { // Refactor convOp
    nextOp = use.getOwner();
    auto convOp = dyn_cast<tpu::Conv2DOp>(nextOp);
    convOp->setAttr("strides",
                    rewriter.getI64ArrayAttr({1, 1})); // rewrite strideH
  }

  int on = shape[0];
  int oc = shape[1];
  int oh = shape[2] / strideH;
  int ow = shape[3] / strideW;
  op->setAttr("do_early_stride", rewriter.getBoolAttr(true));
  op->setAttr("early_stride_h", rewriter.getI32IntegerAttr(strideH));
  op->setAttr("early_stride_w", rewriter.getI32IntegerAttr(strideW));
  auto tensorType = op->getResult(0).getType().cast<RankedTensorType>();
  auto type =
      RankedTensorType::get({on, oc, oh, ow}, tensorType.getElementType());
  op->getResult(0).setType(type); // rewrite inputShape
  return success();
}
} // namespace cv18xx
} // namespace tpu_mlir
