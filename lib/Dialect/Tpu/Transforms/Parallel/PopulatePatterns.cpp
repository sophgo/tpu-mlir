//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Parallel.hpp"

namespace tpu_mlir {
namespace tpu {
using namespace bm1684x;

void populateParalleBM1684XPatterns(RewritePatternSet *patterns, int coreNum){
    // Add an Op-specific pattern if the generic IndexingMap fails to capture
    // the parallel semantics in this operation.
    // patterns->add<
    //   Parallel<tpu::someOp>
    // >(patterns->getContext(), coreNum);
};

} // namespace tpu
} // namespace tpu_mlir
