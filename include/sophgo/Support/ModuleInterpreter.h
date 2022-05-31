//===- ModuleInterpreter.h - Interpreter ------------------------------*- C++
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose interpreter constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MODULEINTERPRETER_H_
#define MLIR_MODULEINTERPRETER_H_

#include "sophgo/Interfaces/InferenceInterface.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <map>

#define DEBUG_TYPE "interpreter"

using namespace mlir;
namespace sophgo {
// Implementation class for module interpreter.
class ModuleInterpreter {

public:
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module);
  virtual ~ModuleInterpreter();
  void allocate_resources();
  void invoke(bool express_type = true);
  void setTensor(const std::string &name, const void *data, size_t size);
  std::shared_ptr<std::vector<float>> getTensor(const std::string &name);
  llvm::ArrayRef<int64_t> getTensorShape(const std::string &name);

public:
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> all_tensor_names; // activation tensor, without weight

private:
  ModuleOp module;
  llvm::StringRef state;
  std::map<std::string, mlir::Value> value_map;
  std::map<std::string, std::shared_ptr<InferenceParameter>> inference_map;
  std::map<std::string, std::shared_ptr<std::vector<float>>> mem_map;
};

} // namespace mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
