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

#include "sophgo/Dialect/Top/Transforms/Passes.h"
#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Support/Helper/Module.h"
#include "sophgo/Support/Helper/Quant.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <sstream>
#include <fstream>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace sophgo::helper;
namespace sophgo {
namespace top {

class SaveWeightPass : public SaveWeightBase<SaveWeightPass> {
public:
  SaveWeightPass() {}
  void runOnOperation() override {
    auto module = getOperation();
    // check name conflict
    std::set<StringRef> all_names;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation* op) {
        if (op->hasAttr("name")) {
          auto name = op->getAttr("name").cast<StringAttr>().getValue();
          if (all_names.find(name) != all_names.end()) {
            op->dump();
            llvm_unreachable("op name conflict");
          }
          all_names.insert(name);
        }
      });
    }
    // weight remove unused in npz
    auto dialect = module->getContext()->getLoadedDialect("top");
    auto top_dialect = llvm::cast<top::TopDialect>(dialect);
    if (top_dialect->wFile == nullptr) {
      return;
    }
    if (top_dialect->wFile->changed() == false) {
      return;
    }
    std::set<StringRef> weight_names;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](top::WeightOp op) { weight_names.insert(op.name()); });
    }
    std::set<StringRef> npz_names;
    top_dialect->wFile->getAllNames(npz_names);
    std::set<StringRef> dif_names;
    for (auto name : npz_names) {
      if (weight_names.find(name) == weight_names.end()) {
        dif_names.insert(name);
      }
    }
    for (auto &name : dif_names) {
      top_dialect->wFile->deleteTensor(name);
    }
    if (top_dialect->wFile->changed() == false) {
      return;
    }
    auto file_name = Module::genWeightFileName(module);
    top_dialect->wFile->save(file_name);
    Module::setWeightFile(module, file_name);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSaveWeightPass() {
  return std::make_unique<SaveWeightPass>();
}
} // namespace top
} // namespace sophgo
