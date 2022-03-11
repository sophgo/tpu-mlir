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
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"
#include <sstream>
#include <fstream>
#include <regex>
#include <unordered_map>

using namespace llvm;

namespace mlir {
namespace tops {

class QuantizePass
    : public QuantizeBase<QuantizePass> {
public:
  QuantizePass() {}
  void runOnOperation() override {
    llvm::errs() << "default quantize mode:" << this->mode
                 << ", is asymmetric " << this->isAsymmetric
                 << ", chip :" << this->chip << "\n";
    auto module = getOperation();
    auto state = getMlirState(module);
    if (state != "TOPS_CALIBRATED" && mode != "INT8") {
      module.dump();
      llvm_unreachable("Mlir state not support quantize");
    }
    for (auto func : getOperation().getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
      });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createQuantizePass() {
  return std::make_unique<QuantizePass>();
}
} // namespace tops
} // namespace mlir
