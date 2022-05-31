//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Top/Transforms/Passes.h"
#include "sophgo/Support/Helper/Module.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
    Module::removeUnusedOp(module);
    // check name conflict
    std::set<StringRef> all_names;
    for (auto func : module.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (op->hasAttr("name")) {
          if (op->getUses().empty()) {
            op->erase();
          } else {
            auto name = Module::getName(op);
            if (all_names.find(name) != all_names.end()) {
              op->dump();
              llvm_unreachable("op name conflict");
            }
            all_names.insert(name);
          }
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
