//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Definitions of common utilities for working with files.
//
//===----------------------------------------------------------------------===//

#include "sophgo/Support/TensorFile.h"

using namespace mlir;

namespace mlir {

std::unique_ptr<TensorFile>
openInputTensorFile(StringRef inputFilename) {
  return std::make_unique<TensorFile>(inputFilename, true, false);
}

std::unique_ptr<TensorFile>
openOutputTensorFile(llvm::StringRef outputFilename) {
  return std::make_unique<TensorFile>(outputFilename, false, true);
}

std::unique_ptr<TensorFile>
openTensorFile(llvm::StringRef filename) {
  return std::make_unique<TensorFile>(filename, false);
}

} // namespace mlir
