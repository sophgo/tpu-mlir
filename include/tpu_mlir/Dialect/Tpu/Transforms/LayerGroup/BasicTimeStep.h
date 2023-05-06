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
#include "tpu_mlir/Support/Module.h"
#include <list>
#include <map>
#include <set>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/SwPipeline.h"

namespace tpu_mlir {
namespace tpu {

class TimeStepMethod;

class BasicTimeStep {
public:
  BasicTimeStep();
  virtual ~BasicTimeStep() {}
  void clear();

  bool assignTimeStep(const LgInfo &lg_info, const shape_secs_t &shape_secs,
                      bool gen_idx);
  void add_tpu0_ts_field(const TpuTsField &field);
  void add_gdma0_ts_field(const GdmaTsField &field);
  void add_tpu0_gdma0_ts_field(const TpuTsField &tpu_field,
                               const GdmaTsField &gdma_field);
  void update_gdma0_ts_field(int64_t ts, const GdmaTsField &field);
  std::vector<TimestepRow> &get_timestep_table() { return timestep_table_; }
  size_t get_timestep_num() { return timestep_table_.size(); }

  int64_t get_tensor_life_time(Value v);

  int64_t get_tensor_range_end(const GdmaElt &tensor, int64_t cur_ts);
  std::shared_ptr<SoftwarePipeline> get_timestep_swpipl() { return swpipl_; }
  int64_t get_layer_swpipl_stage(Operation *op);
  int64_t get_tensor_swpipl_stage(Value v);
  int64_t get_swpipl_stage_num() { return swpipl_stage_num_; }
  void set_swpipl_stage_num(int num) {
    swpipl_stage_num_ = num;
  } // just for ir gen
  void software_pipeline();

  // getter
  TpuTsField &getLayers(int64_t ts) {
    return timestep_table_[ts].tpu0_ts_field;
  }
  GdmaTsField &getTensors(int64_t ts) {
    return timestep_table_[ts].gdma0_ts_field;
  }
  MemBlock find_buffer_locate(Value value, int64_t ts, const MemBuff &buffer, const MemBuff &l2buffer);
  int64_t get_lmem_addr(const mem_buffer_key_t &buffer_key);
  int64_t get_lmem_size(const mem_buffer_key_t &buffer_key);
  MemBlock get_lmem_locate(Value value, int64_t ts);
  MemBuff &get_lmem_buffer() { return lmem_buffer_; }
  MemBuff &get_l2mem_buffer() { return l2mem_buffer_; }

  const mem_buffer_value_t &
  get_lmem_buffer_value(const mem_buffer_key_t &buffer_key);
  const mem_buffer_value_t &
  get_l2mem_buffer_value(const mem_buffer_key_t &buffer_key);
  int64_t get_lmem_occupy() const { return lmem_occupy_; }
  std::map<Value, int64_t, value_compare> &get_hold_coeff() {
    return hold_coeff_;
  }

  TensorInfo &get_tensor_infos();

  // setter
  void set_lmem_addr(const mem_buffer_key_t &buffer_key, int64_t lmem_addr);
  void set_lmem_occupy(int64_t occupy) { lmem_occupy_ = occupy; }

  void gen_all_mem_buffer();
  void update_all_mem_buffer_size(const LgInfo &lg_info);
  void gen_hold_coeff();
  bool is_tensor_hold_in_lmem(Value v);
  void cancel_tensor_hold_in_lmem(Value v);

  // visualizer
  void show_timestep();
  void show_lmem_buffer();

  bool layer_can_merge_backward(int64_t ts, bool consider_hold_in_coeff);

  //==============================================
  // functions for timestep combine
  //==============================================
  void reset_timestep(std::vector<TpuTsField> &ts_layers_v,
                      std::vector<GdmaTsField> &ts_tensors_v,
                      MemBuff &mem_buffer);
  void clear_gdma_cycle() { gdma_cycle_count_.clear(); }
  void clear_layer_cycle() { layer_cycle_count_.clear(); }
  void set_gdma_cycle(Value value, int64_t cycle_count) {
    gdma_cycle_count_[value] = cycle_count;
  }
  void set_layer_cycle(Operation *op, int64_t cycle_count) {
    layer_cycle_count_[op] = cycle_count;
  }
  ValueIntMap &get_gdma_cycle() { return gdma_cycle_count_; }
  std::map<Operation *, int64_t> &get_layer_cycle() {
    return layer_cycle_count_;
  }
  bool tensor_can_move(GdmaElt &tensor, int64_t src_ts, int64_t dst_ts);

  //==============================================
  // functions for group overlap
  //==============================================
  void insert_self_up_op(Value value);
  void insert_self_down_op(Value value);
  void insert_other_up_op(Value value, int64_t dst_ts);
  void insert_other_down_op(Value value, int64_t dst_ts);
  ValueSet &get_self_up_overlap_ops() { return self_up_overlap_ops_; }
  ValueSet &get_self_down_overlap_ops() { return self_down_overlap_ops_; }
  std::map<int64_t, std::vector<Value>> &get_other_up_overlap_ops() {
    return other_up_overlap_ops_;
  }
  std::map<int64_t, std::vector<Value>> &get_other_down_overlap_ops() {
    return other_down_overlap_ops_;
  }

protected:
  std::shared_ptr<TimeStepMethod> timestep_method_;
  std::shared_ptr<SoftwarePipeline> swpipl_;
  std::vector<TimestepRow> timestep_table_;

  ValueIntMap hold_coeff_;
  ValueIntMap canceled_hold_coeff_;
  TensorInfo tensor_infos_;
  int64_t swpipl_stage_num_;

  int64_t lmem_occupy_;
  MemBuff lmem_buffer_;
  MemBuff l2mem_buffer_;

  // members for timestep combine
  ValueIntMap gdma_cycle_count_;
  std::map<Operation *, int64_t> layer_cycle_count_;

  // used for group overlap
  ValueSet self_up_overlap_ops_;
  ValueSet self_down_overlap_ops_;
  std::map<int64_t, std::vector<Value>> other_up_overlap_ops_;
  std::map<int64_t, std::vector<Value>> other_down_overlap_ops_;
};

using BasicTimeStepPtr = std::shared_ptr<BasicTimeStep>;

} // namespace tpu
} // namespace tpu_mlir
