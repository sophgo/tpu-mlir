//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
using namespace tpu_mlir::tpu;

namespace tpu_mlir {
namespace tpu {

SoftwarePipeline::SoftwarePipeline() { clear_all(); }

void SoftwarePipeline::clear_all() { tensor_swloop_buffer_.clear(); }

void SoftwarePipeline::clear_swloop_buffer() { tensor_swloop_buffer_.clear(); }

void SoftwarePipeline::write_swloop_buffer(int64_t nstep, int64_t cstep,
                                           int64_t hstep, int64_t dstep,
                                           int64_t wstep, int64_t stage_num) {
  // add swloop buffer
  tensor_step_t tensor_step = {nstep, cstep, hstep, dstep, wstep};
  tensor_swloop_buffer_.push_front(tensor_step);
  while (tensor_swloop_buffer_.size() > (uint32_t)stage_num) {
    tensor_swloop_buffer_.pop_back();
  }
}

const tensor_step_t *SoftwarePipeline::read_swloop_buffer(int64_t stage) {
  std::list<tensor_step_t>::iterator iter = tensor_swloop_buffer_.begin();
  for (int i = 0; i < stage; ++i) {
    iter++;
  }
  return &(*iter);
}

int64_t SoftwarePipeline::get_tensor_swpipl_stage(Value v) {
  auto iter = tensor_swpipl_stage_.find(v);
  if (iter != tensor_swpipl_stage_.end()) {
    return iter->second;
  }
  return 1;
}

int64_t SoftwarePipeline::software_pipeline_schedule(
    std::vector<TimestepRow> &timestep_table) {
  // assert the first and the last timestep only contain tensor gdma
  auto first_row_iter = timestep_table.begin();
  if (!(first_row_iter->tpu0_ts_field.empty())) {
    llvm::errs() << "simple software pipeline schedule assume the first "
                    "timestep only contains tensor gdma.";
    exit(-1);
  }

  auto last_row_iter = timestep_table.end();
  last_row_iter--;
  if (!(last_row_iter->tpu0_ts_field.empty())) {
    llvm::errs() << "simple software pipeline schedule assume the last "
                    "timestep only contains tensor gdma.";
    exit(-1);
  }

  //==============================
  // stage assignment
  //==============================
  // 3-stage software pipeline default
  int64_t stage_num = 3;
  // layers are assigned to stage 1 default
  for (uint32_t i = 0; i < timestep_table.size(); ++i) {
    const GdmaTsField &tensors = timestep_table[i].gdma0_ts_field;
    for (uint32_t j = 0; j < tensors.size(); ++j) {
      if (i == 0) {
        // assign stage 0 for tensors in the first timestep
        tensor_swpipl_stage_.insert(std::make_pair(tensors[j].first, 0));
      } else if (i == timestep_table.size() - 1) {
        // assign stage 2 for tensors in the last timestep
        tensor_swpipl_stage_.insert(std::make_pair(tensors[j].first, 2));
      } else {
        // assign stage 1 for other tensors
        tensor_swpipl_stage_.insert(std::make_pair(tensors[j].first, 1));
      }
    }
  }

  //=============================
  // stage task schedule
  //=============================
  GdmaTsField first_tensor_timestep = first_row_iter->gdma0_ts_field;
  GdmaTsField last_tensor_timestep = last_row_iter->gdma0_ts_field;

  // delete the last row of time step table
  timestep_table.erase(last_row_iter);

  // 1. (try) move the last tensor timestep to the first
  bool move_valid;
  // consider time step 1, it is the second row of the table
  auto second_row_iter = timestep_table.begin() + 1;

  GdmaTsField rest_last_tensors_;
  for (uint32_t i = 0; i < last_tensor_timestep.size(); ++i) {
    move_valid = true;
    auto v = last_tensor_timestep[i].first;
    // if v is used by tpu op (is opds or results of tpu op) in the second row
    // it cannot be moved from last to second row (move_valid = false)
    for (auto op : second_row_iter->tpu0_ts_field) {
      auto opds = op->getOperands();
      auto results = get_output_values(op);
      if (std::find(opds.begin(), opds.end(), v) != opds.end() ||
          std::find(results.begin(), results.end(), v) != results.end()) {
        move_valid = false;
        rest_last_tensors_.push_back(last_tensor_timestep[i]);
        break;
      }
    }

    if (move_valid) {
      second_row_iter->gdma0_ts_field.push_back(last_tensor_timestep[i]);
    }
  }

  if (rest_last_tensors_.empty()) {
    // delete the first time step
    timestep_table.erase(timestep_table.begin());
  } else {
    // remain the first time step
    timestep_table[0].gdma0_ts_field.clear();
    timestep_table[0].gdma0_ts_field = rest_last_tensors_;
  }

  // 2. (try) move the first tensor timestep to the last
  last_row_iter = timestep_table.end() - 1;
  // consider time step n-1, it is the (n-1)th row of the table
  GdmaTsField rest_first_tensors_;
  for (uint32_t i = 0; i < first_tensor_timestep.size(); ++i) {
    move_valid = true;
    auto v = first_tensor_timestep[i].first;

    // if v is used by tpu op (is opds or results of tpu op) in the "new" last
    // row it cannot be moved from first to the "new" last row (move_valid =
    // false) note: the "new" last row is the last row before the "real" last
    // row which is deleted
    for (auto op : last_row_iter->tpu0_ts_field) {
      auto opds = op->getOperands();
      if (std::find(opds.begin(), opds.end(), v) != opds.end()) {
        move_valid = false;
        rest_first_tensors_.push_back(first_tensor_timestep[i]);
        break;
      }
    }

    if (move_valid) {
      last_row_iter->gdma0_ts_field.push_back(first_tensor_timestep[i]);
    }
  }

  if (!rest_first_tensors_.empty()) {
    TimestepRow new_row;
    new_row.gdma0_ts_field = rest_first_tensors_;
    timestep_table.push_back(new_row);
  }
  return stage_num;
}

} // namespace tpu
} // namespace tpu_mlir
