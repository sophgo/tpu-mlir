//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the TPU dialect OP Stats pass.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tops/Transforms/Passes.h"
#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <sstream>
#include <fstream>
#include <regex>
#include <unordered_map>

#define DEBUG_TYPE "import_calibration_table"

using namespace llvm;

namespace mlir {
namespace tops {
class ImportCalibrationPass
    : public ImportCalibrationBase<ImportCalibrationPass> {
public:
  ImportCalibrationPass() {}
  void runOnOperation() override {
    llvm::errs() << "import calibration table:" << this->calibrationTable << "\n";
    for (auto func : getOperation().getOps<FuncOp>()) {
    }
  }

private:
  StringRef table;
};

std::unique_ptr<OperationPass<ModuleOp>>
createImportCalibrationPass() {
  return std::make_unique<ImportCalibrationPass>();
}
}
} // namespace mlir
