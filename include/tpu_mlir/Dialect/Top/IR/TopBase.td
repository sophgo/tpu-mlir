//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TOP_BASE
#define TPU_MLIR_TOP_BASE

include "mlir/IR/OpBase.td"

def Top_Dialect : Dialect {
  let name = "top";
  let summary = "A topdialect for the TPU_MLIR specification";
  let cppNamespace = "::tpu_mlir::top";
  let extraClassDeclaration = [{
  void loadWeightFile(llvm::StringRef file) {
    wFile = std::make_unique<mlir::TensorFile>(file, false);
  }
  std::unique_ptr<mlir::TensorFile> wFile;
  }];
}

#endif
