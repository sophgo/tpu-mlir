//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/ExtraConversion/DoExtraConversion.h"
#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertBM1684.h"
#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertBM1684X.h"
#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"
#include "tpu_mlir/Conversion/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "tpu_mlir/Backend/Arch.h"

#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_DOEXTRACONVERSION
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

struct DoExtraConversion
    : public ::impl::DoExtraConversionBase<DoExtraConversion> {
public:
  DoExtraConversion() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    RewritePatternSet patterns(mOp.getContext());
    if (module::isBM1684XFamily() || module::isBM1686()) {
      bm1684x::populateDoExtraConversionPatterns(&patterns);
    } else if (module::isBM1684Family()) {
      bm1684::populateDoExtraConversionPatterns(&patterns);
    } else if (module::isCV18xx()) {
      cv18xx::populateDoExtraConversionPatterns(&patterns);
    }
    auto config = GreedyRewriteConfig();
    config.maxIterations = 5; // apply each pattern only once.
    applyPatternsAndFoldGreedily(mOp, std::move(patterns), config);
    module::updateModuleTypes();
  }
};

std::unique_ptr<Pass> createDoExtraConversion() {
  return std::make_unique<DoExtraConversion>();
}
} // namespace tpu_mlir
