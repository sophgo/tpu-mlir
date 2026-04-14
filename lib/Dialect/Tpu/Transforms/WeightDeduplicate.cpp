//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "llvm/Support/MD5.h"
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

static std::string calcWeightKey(top::WeightOp op) {
  auto type = op.getOutput().getType().cast<RankedTensorType>();
  auto shape = type.getShape();
  auto elemType = type.getElementType();
  auto bytes = op.read_as_byte();

  llvm::MD5 md5;
  md5.update(ArrayRef<uint8_t>(bytes->data(), bytes->size()));
  llvm::MD5::MD5Result result;
  md5.final(result);
  SmallString<32> md5Str;
  llvm::MD5::stringifyResult(result, md5Str);

  std::string key;
  llvm::raw_string_ostream os(key);
  os << elemType;
  os << "|shape=";
  for (auto d : shape) {
    os << d << ",";
  }
  os << "|store_mode=";
  if (op.getStoreMode().has_value()) {
    os << op.getStoreModeAttr().getValue();
  } else {
    os << "none";
  }
  os << "|bytes=" << md5Str;
  return os.str();
}

class WeightDeduplicatePass
    : public WeightDeduplicateBase<WeightDeduplicatePass> {
public:
  WeightDeduplicatePass() {}

  void runOnOperation() override {
    auto modules = module::getAllModules();

    for (auto sub : *modules) {
      std::unordered_map<std::string, top::WeightOp> firstWeight;
      std::vector<top::WeightOp> duplicated;

      for (auto func : sub.getOps<FuncOp>()) {
        func.walk([&](top::WeightOp w) {
          if (w.use_empty()) {
            return;
          }

          if (w->hasAttr("allow_split") || w->hasAttr("indices_idx") ||
              w->hasAttr("indices_slice")) {
            return;
          }

          auto key = calcWeightKey(w);
          auto it = firstWeight.find(key);
          if (it == firstWeight.end()) {
            firstWeight.emplace(key, w);
            return;
          }

          auto canonical = it->second;
          if (canonical == w) {
            return;
          }

          w.getOutput().replaceAllUsesWith(canonical.getOutput());
          duplicated.push_back(w);
        });
      }

      for (auto w : duplicated) {
        if (w.use_empty()) {
          w.erase();
        }
      }
    }

    module::removeUnusedOp();
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createWeightDeduplicatePass() {
  return std::make_unique<WeightDeduplicatePass>();
}

} // namespace tpu
} // namespace tpu_mlir
