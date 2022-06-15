//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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
