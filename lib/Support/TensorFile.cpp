//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// Definitions of common utilities for working with files.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/TensorFile.h"

using namespace mlir;

namespace mlir {

std::unique_ptr<TensorFile> openInputTensorFile(StringRef inputFilename) {
  return std::make_unique<TensorFile>(inputFilename, true, false);
}

std::unique_ptr<TensorFile>
openOutputTensorFile(llvm::StringRef outputFilename) {
  return std::make_unique<TensorFile>(outputFilename, false, true);
}

std::unique_ptr<TensorFile> openTensorFile(llvm::StringRef filename) {
  return std::make_unique<TensorFile>(filename, false);
}

} // namespace mlir
