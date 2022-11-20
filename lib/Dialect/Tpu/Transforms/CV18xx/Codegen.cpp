//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/MlirToCvimodel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;
using namespace flatbuffers;
namespace tpu_mlir {
namespace tpu {

class CVCodegenPass : public CVCodegenBase<CVCodegenPass> {
public:
  CVCodegenPass() {}
  void runOnOperation() override {
    module = getOperation();
    state = Module::getState(module);
    chip = Module::getChip(module);
    assert(state == Module::State::TPU_ADDRESSED);
    assert(chip == Module::Chip::CV182x || chip == Module::Chip::CV183x);
    std::string filename = this->model_file;
    if (filename.empty()) {
      llvm_unreachable("output filename is empty");
    }
    Arch::init(chip);
    CviModelBuilder builder(module);
    builder.storeModel(filename);
  }

private:
  ModuleOp module;
  StringRef state;
  StringRef chip;
};

std::unique_ptr<OperationPass<ModuleOp>> createCVCodegenPass() {
  return std::make_unique<CVCodegenPass>();
}

} // namespace tpu
} // namespace tpu_mlir
