//===- FileUtilities.cpp - utilities for working with files ---------------===//
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
