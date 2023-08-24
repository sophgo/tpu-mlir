//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace tpu {
namespace bm1684x {

template <typename Op>
class Parallel : public OpRewritePattern<Op> {
public:
  Parallel(MLIRContext *context, int coreNum = 1)
      : coreNum(coreNum), OpRewritePattern<Op>(context){};

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override;

private:
  int coreNum;
};

} // namespace bm1684x
} // namespace tpu
} // namespace tpu_mlir
