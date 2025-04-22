//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Codegen/BM168xCodegen.hpp"
#include "Codegen/CV18xxCodegen.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

class CodegenPass : public CodegenBase<CodegenPass> {
public:
  CodegenPass() {}
  void runOnOperation() override {
    assert(module::isState(module::State::TPU_ADDRESSED));
    std::string filename = this->model_file;
    if (filename.empty()) {
      llvm_unreachable("output filename is empty");
    }
    auto mOp = getOperation();
    auto modules = module::getAllModules();
    if (module::isCV18xx()) {
      CviModelBuilder builder(modules->at(0), model_version);
      builder.storeModel(filename);
      return;
    }
    BMCodegen bm_codegen;
    bm_codegen.init(mOp, filename, bmodel_only);
    int num_device = module::getDeviceNum();
    int num_submodule = module::getNumSubModule();
    if (num_device > num_submodule) {
      assert(num_submodule == 1);
      auto sub_m = modules->at(0);
      auto name = module::getName(sub_m).str();
      for (int i = 0; i < num_device; ++i) {
        auto new_name = name + "_" + std::to_string(i);
        sub_m.setName(new_name);
        module::setSubModuleId(sub_m, i, 0);
        bm_codegen.run(sub_m, embed_debug_info, gdma_check);
      }
    } else {
      for (auto s : *modules) {
        bm_codegen.run(s, embed_debug_info, gdma_check);
      }
    }

    bm_codegen.store();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass() {
  return std::make_unique<CodegenPass>();
}

} // namespace tpu
} // namespace tpu_mlir
