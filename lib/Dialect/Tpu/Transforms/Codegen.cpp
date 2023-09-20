//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Codegen/CV18xxCodegen.hpp"
#include "Codegen/BM168xCodegen.hpp"
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
    bm_codegen.init(mOp, filename);
    for (auto s : *modules) {
      bm_codegen.run(s, embed_debug_info);
    }
    bm_codegen.store();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCodegenPass() {
  return std::make_unique<CodegenPass>();
}

} // namespace tpu
} // namespace tpu_mlir
