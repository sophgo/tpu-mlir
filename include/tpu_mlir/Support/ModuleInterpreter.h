//===- ModuleInterpreter.h - Interpreter ------------------------------*- C++
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose interpreter constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MODULEINTERPRETER_H_
#define MLIR_MODULEINTERPRETER_H_

#include "tpu_mlir/Interfaces/InferenceInterface.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Debug.h"

#include <fstream>
#include <iostream>
#include <map>

#define DEBUG_TYPE "interpreter"

using namespace mlir;
namespace tpu_mlir {
// Implementation class for module interpreter.
class ModuleInterpreter {
public:
  enum class mem_mode_t {
    // if mem size > 16GB, then use ALL_TENSOR_IN_DISK or PART_TENSOR_IN_MEM
    // else use ALL_TENSOR_IN_MEM
    ALL_TENSOR_IN_MEM,
    ALL_TENSOR_IN_DISK,
    PART_TENSOR_IN_MEM
  };
  // Interpret the given MLIR module expressed in MLIR TPU IR dialect
  explicit ModuleInterpreter(ModuleOp module);
  virtual ~ModuleInterpreter();
  void allocate_resources();
  void invoke(bool express_type = true);
  void invoke_to_disk(const std::string &filename, bool express_type = true);
  void fake_quant_weight();
  std::shared_ptr<std::vector<float>> invoke_at(std::string name);
  void invoke_from(const std::string op_name);
  void setTensor(const std::string &name, const void *data, size_t size,
                 bool is_integer = false);
  std::shared_ptr<std::vector<float>> getTensor(const std::string &name,
                                                bool express_type = false);
  bool getTensorQuantInfo(const std::string name, std::string &dtype,
                          float &scale, int &zp);
  llvm::ArrayRef<int64_t> getTensorShape(const std::string &name);
  bool is_no_mem_op(Operation *op);

private:
  void allocate_part_tensor_in_mem();
  void allocate_all_tensor_in_mem();
  void allocate_all_tensor_in_disk();
  bool check_op_in_mem(Operation *op);
  void invoke_part_in_mem(bool express_type = true);
  void invoke_all_in_mem(bool express_type = true);
  void value_to_disk(const std::string &filename, const std::string &name,
                     std::vector<float> &data, bool express_type = true);
  void collect_tensor(Value v);

public:
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string>
      all_tensor_names; // activation tensor, without weight
  std::vector<std::string> all_weight_names; // weight tensor

private:
  ModuleOp module;
  int64_t num_infer_op;
  mem_mode_t mem_mode;
  int64_t total_count;
  std::map<std::string, Value> value_map;
  std::map<std::string, std::shared_ptr<InferenceParameter>> inference_map;
  std::map<std::string, std::shared_ptr<std::vector<float>>> mem_map;
};

} // namespace tpu_mlir

#endif // MLIR_DIALECT_TPU_MODULEINTERPRETER_H_
