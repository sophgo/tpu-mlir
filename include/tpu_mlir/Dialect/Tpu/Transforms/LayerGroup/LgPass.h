//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/IlpTimeStep.h"

namespace tpu_mlir {
namespace tpu {

enum class NnvlcMode {
    NONE = 0,
    WEIGHT = 1,
    ACTIVATION = 2,
    ALL = 3
};

typedef struct {
  bool dyn_compile;
  int64_t opt;
  bool group_by_cores;
  NnvlcMode nnvlc_mode;
  bool lgcache;
} LgOptions;

// struct node_info;
struct LgPassIR {
  LgPassIR(){returnOp = nullptr;};
  ~LgPassIR() { clear(); };

  /**
   * Clear information of layer group IR
   */
  void clear();

  /**
   * @brief the operation in the current subnet graph
   */
  llvm::SetVector<Operation *> subnet_ops;

  /**
   * @brief the value in the current subnet graph
   */
  llvm::SetVector<Value> subnet_values;

  /**
   * @brief the layer groups.
   * lg_infos.size() means the number of groups.
   * lg_infos[i].ids.size() means the number of layers in the i-th group
   */
  std::vector<LgInfo> lg_infos;

  /**
   * @brief time step of layer groups
   * time_steps.size() == lg_infos.size()
   * time_steps[i] means the time step of the i-th group
   */
  std::vector<BasicTimeStepPtr> time_steps;
  std::vector<std::vector<ILPTimeStepPtr>> ILP_time_steps;
  std::vector<std::map<int, std::vector<l2m_value_info>>> map_l2m_loads;
  std::vector<TensorInfo> lg_tensor_infos_;
  std::vector<l2mem_alloc_Ptr> lg_l2mem_alloc_ptr;
  std::vector<int> group_cycles;

  /**
   * @brief shape split sections of layer groups
   * shape_secs.size() == lg_infos.size()
   * shape_secs[i] means the shape split sections of the i-th group
   */
  std::vector<shape_secs_t> shape_secs;

  std::vector<Value> subnet_return_opds;
  bool branch_parallel;
  std::vector<std::shared_ptr<ilp_LgInfo>> tmp_base_groups;
  std::map<Operation*, std::vector<Operation*>> map_parallel_op_subnet;
  Operation* returnOp;
  std::shared_ptr<dot_graph> dot_graph_log_subnet;
  FuncOp func;
};

class LgPass {
public:
  LgPass() {}
  virtual ~LgPass() {}

  virtual bool run(LgPassIR *pass_ir) = 0;
  virtual std::string name() = 0;
  virtual std::string brief() { return ""; }
  static LgOptions OPTIONS; // global options
};

/// Pass manager of layer group optimization
class LgPassManager {
public:
  LgPassManager() {}
  ~LgPassManager() {}

  void add_pass(std::unique_ptr<LgPass> pass);
  void run(LgPassIR *pass_ir);

private:
  std::vector<std::unique_ptr<LgPass>> passes;
};

/// Layer group optimizer
class LgOptimizer {
public:
  LgOptimizer() {}
  virtual ~LgOptimizer() {}

  virtual void manage_passes(std::shared_ptr<LgPassManager> pm,
                             const LgOptions &options) = 0;
  virtual std::string brief() = 0;
};

using LgOptimizerMap = std::map<std::string, LgOptimizer *>;

const LgOptimizerMap &get_registered_optimizers();

struct LgOptimizerReg {
  LgOptimizerReg(const std::string &name,
                 std::shared_ptr<LgOptimizer> optimizer);
};

#define REGISTER_LG_OPTIMIZER(name, optimizer)                                 \
  static std::shared_ptr<LgOptimizer> name##_opt_inst(new optimizer());        \
  static LgOptimizerReg name##_lg_reg(#name, name##_opt_inst)

} // namespace tpu
} // namespace tpu_mlir
