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
  GroupOps(::mlir::func::FuncOp func, int64_t opt);
  ~GroupOps() { delete lg_pass_ir_; }
  void process(int64_t opt);
  ::mlir::func::FuncOp func_;

protected:
  // create groups
  void buildGroups(int64_t opt);
  //  void assign_timestep();
  //  bool assign_lmem_addr();
  //nnvlc
  void buildNnvlcActivation();

  // create MLIR GroupOp
  void buildMlir();
  void buildMlir_for_opt3();
  void buildGroupOp(const LgInfo &lg_info, const shape_secs_t &shape_secs,
                    int64_t group_idx);
  void CreateLoadOp(GdmaElt &tensor, int64_t id,
                    const std::vector<Operation *> &ops,
                    group_type_t group_type);

  tpu::StoreOp CreateStoreOp(GdmaElt &tensor, int64_t id,
                             group_type_t group_type);
  void UpdateGroupOverlapInfo(Operation *op, int64_t group_idx);
  void UpdateOpLgParam(Operation *op, TensorInfo &tensor_infos, int64_t id,
                       group_type_t group_type);
  tpu::LayerGroupAttr getLgParam(tensor_info_t &tensor_info, int64_t id,
                                 int64_t out_addr, int64_t out_size,
                                 int64_t group_type = 0,
                                 int64_t buffer_addr = 0,
                                 int64_t buffer_size = 0, int64_t slice_idx = 0, bool can_merge = false,
                                 std::vector<std::vector<int64_t>> opd_h_slice_offset = {});
  //  bool need_none(group_lmem_t &group_lmem);
  void CreateLmemMoveOp(int64_t ts, ts_move_info& move_info);
  void CreateLoadOp2(int64_t ts, ts_var_t& ts_var, int64_t pipe_id,
                     const std::vector<Operation *> &ops, std::vector<int64_t> ncdhw_idx,
                     const LgInfo& lgInfo, bool can_merge);
  void CreateLoadToL2mOp(int64_t ts, l2m_value_info& it, int64_t pipe_id, l2mem_alloc_Ptr l2mem_alloc_ptr);
  Value CreateStoreOp2(Value &output, tensor_info_t& ti, int64_t ts, int64_t slice_idx, int64_t pipe_id,
                       group_type_t group_type, bool can_merge);
  void UpdateOpLgParam2(Operation *op, Operation *old_op, int64_t ts, int64_t slice_idx, TensorInfo &tensor_info, std::vector<int64_t> ncdhw_idx,
                       group_type_t group_type, bool can_merge);
  void find_local_layer_base_group(Operation * op);

protected:
  std::shared_ptr<GroupMethod> group_method_;
  std::vector<BasicTimeStepPtr> time_steps_;
  std::vector<LgInfo> lg_infos_;
  BasicTimeStepPtr time_step;
  LgPassIR *lg_pass_ir_;

  mlir::MLIRContext *ctx_;
  Operation *current_op_;
  Block *body_;
  int64_t MAX_ID_;

  // used for group overlap
  IntValueIntMap self_up_overlap_ops_;
  IntValueIntMap self_down_overlap_ops_;
  std::vector<Operation *> groups_;
  ILPTimeStepPtr ILP_time_step;
  std::map<Value, std::map<int, Value>, value_compare> map_old_to_new_value;
  int64_t version;
  std::map<Value, std::vector<std::string>, value_compare> map_name_output_to_merge_slice_for_grp;
  std::map<Value, Value, value_compare> map_store_tensor_to_outbuffer_out;
  std::map<Value, Value, value_compare> map_old_grp_out_to_new_grp_out;
  std::vector<std::vector<Value>> need_store_load_value;
  std::map<Value, Value, value_compare> map_store_to_load_value;
  std::map<Value, Value, value_compare> map_l2m_out_to_load_in;
  std::vector<std::vector<Value>> will_store_value;
  std::vector<Operation*> tmp_local_layer_group;
  std::vector<Operation*> all_local_layer_nodes;
  bool branch_parallel = false;
};

} // namespace tpu
} // namespace tpu_mlir
