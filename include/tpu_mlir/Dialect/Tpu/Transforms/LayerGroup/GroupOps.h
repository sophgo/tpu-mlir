//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Support/LLVM.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>
namespace tpu_mlir {
namespace tpu {

class GroupOps {
public:
  GroupOps(::mlir::func::FuncOp func);
  ~GroupOps() { delete lg_pass_ir_; }
  void process(int64_t opt);
  ::mlir::func::FuncOp func_;

protected:
  // create groups
  void buildGroups(int64_t opt);
  //  void assign_timestep();
  //  bool assign_lmem_addr();

  // create MLIR GroupOp
  void buildMlir();
  void buildGroupOp(const LgInfo &lg_info, const shape_secs_t &shape_secs);
  void CreateLoadOp(GdmaElt &tensor, int64_t id,
                    const std::vector<Operation *> &ops, group_type_t group_type);

  tpu::StoreOp CreateStoreOp(GdmaElt &tensor, int64_t id, group_type_t group_type);
  void UpdateGroupOverlapInfo(Operation *op);
  void UpdateOpLgParam(Operation *op, TensorInfo &tensor_infos, int64_t id,
                       group_type_t group_type);
  tpu::LayerGroupAttr getLgParam(tensor_info_t &tensor_info, int64_t id,
                                 int64_t out_addr, int64_t out_size,
                                 int64_t group_type = 0,
                                 int64_t buffer_addr = 0,
                                 int64_t buffer_size = 0);
  //  bool need_none(group_lmem_t &group_lmem);

protected:
  std::shared_ptr<GroupMethod> group_method_;
  std::vector<BasicTimeStepPtr> time_steps_;
  std::vector<LgInfo> lg_infos_;
  BasicTimeStepPtr time_step;
  LgPassIR *lg_pass_ir_;

  std::shared_ptr<std::vector<Operation *>> group_ops_;
  std::vector<std::shared_ptr<std::vector<mlir::Operation *>>> groups_ops_;
  std::vector<Operation *> all_ops_;
  std::vector<Value> all_tensors_;
  mlir::MLIRContext *ctx_;
  Operation *current_op_;
  Block *body_;
  int64_t MAX_ID_;

  // used for group overlap
  ValueIntMap overlap_ops_;
  std::vector<Operation*> groups_;
};

} // namespace tpu
} // namespace tpu_mlir
