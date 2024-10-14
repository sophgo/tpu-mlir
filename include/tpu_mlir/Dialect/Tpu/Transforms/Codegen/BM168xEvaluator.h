//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace llvm;

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

class BM168xEvaluator {
public:
  explicit BM168xEvaluator(ModuleOp module);
  void allocate_resources();
  void setTensor(const std::string &name, const void *data, size_t size,
                 bool is_integer = false);
  void invoke();
  std::shared_ptr<std::vector<float>> getTensor(const std::string &name);
  llvm::ArrayRef<int64_t> getTensorShape(const std::string &name);

private:
  void staging_results(GlobalGenInterface& op);
  void staging_results(LocalGenInterface& op, local_sec_info_t sec_info);
  void visit_subnet(func::FuncOp funcOp, int subnet_id);
  void visit_static_subnet(func::FuncOp funcOp, int subnet_id);
  void visit_group_body(GroupOp gOp, Operation *prev_op,
                         Operation *next_op);
  void handle_group_overlap(
      std::map<int64_t, std::vector<Operation *>> cur_other_downs,
      std::map<int64_t, std::vector<Operation *>> cur_other_ups,
      Operation *prev_op, Operation *next_op, int64_t cur_ts,
      bool first_compute_loop, bool last_compute_loop);

public:
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> all_tensor_names; // activation tensor, without weight
  std::vector<std::string> all_weight_names; // weight tensor

private:
  typedef std::vector<int8_t> staging_mem_t;
  ModuleOp module;
  BM168x *bm168x;
  std::vector<int> num_subnet_ops; // used for progress bar
  std::map<std::string, mlir::Value> value_map;
  std::map<std::string, std::shared_ptr<staging_mem_t>> mem_map;
};

} // namespace tpu
} // namespace tpu_mlir
