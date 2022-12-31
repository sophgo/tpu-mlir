//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStep.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::tpu;

SoftwarePipeline::SoftwarePipeline() { clear_all(); }

void SoftwarePipeline::clear_all() { tensor_swloop_buffer_.clear(); }

void SoftwarePipeline::clear_swloop_buffer() { tensor_swloop_buffer_.clear(); }

void SoftwarePipeline::write_swloop_buffer(int64_t nstep, int64_t hstep,
                                           int stage_num) {
  // add swloop buffer
  tensor_step_t tensor_step = {nstep, hstep};
  tensor_swloop_buffer_.push_front(tensor_step);
  while (tensor_swloop_buffer_.size() > (uint32_t)stage_num) {
    std::list<tensor_step_t>::iterator iter = tensor_swloop_buffer_.end();
    iter--;
    tensor_swloop_buffer_.erase(iter);
  }
}

const tensor_step_t* SoftwarePipeline::read_swloop_buffer(int stage) {
  std::list<tensor_step_t>::iterator iter = tensor_swloop_buffer_.begin();
  for(int i = 0; i < stage; ++i) {
    iter++;
  }
  return &(*iter);
}

int SoftwarePipeline::get_tensor_swpipl_stage(Value v) {
  auto iter = tensor_swpipl_stage.find(v);
  if (iter != tensor_swpipl_stage.end()) {
    return iter->second;
  }
  return 1;
}

int SoftwarePipeline::software_pipeline_schedule(
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
  int stage_num = 3;
  // layers are assigned to stage 1 default
  for (uint32_t i = 0; i < timestep_table.size(); ++i) {
    const GdmaTsField &tensors = timestep_table[i].gdma0_ts_field;
    for (uint32_t j = 0; j < tensors.size(); ++j) {
      if (i == 0) {
        // assign stage 0 for tensors in the first timestep
        tensor_swpipl_stage.insert(std::make_pair(tensors[j], 0));
      } else if (i == timestep_table.size() - 1) {
        // assign stage 2 for tensors in the last timestep
        tensor_swpipl_stage.insert(std::make_pair(tensors[j], 2));
      } else {
        // assign stage 1 for other tensors
        tensor_swpipl_stage.insert(std::make_pair(tensors[j], 1));
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
  // move the last tensor timestep to the first
  bool move_valid;
  // consider time step 1, it is the second row of the table
  auto second_row_iter = timestep_table.begin() + 1;

  GdmaTsField rest_last_tensors_;
  for (uint32_t i = 0; i < last_tensor_timestep.size(); ++i) {
    move_valid = true;
    auto v = last_tensor_timestep[i];
    for (auto op : second_row_iter->tpu0_ts_field) {
      auto opds = op->getOperands();
      auto results = op->getResults();
      if (std::find(opds.begin(), opds.end(), v) != opds.end() ||
          std::find(results.begin(), results.end(), v) != results.end()) {
        move_valid = false;
        rest_last_tensors_.push_back(v);
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

  // move the first tensor timestep to the last
  last_row_iter = timestep_table.end() - 1;
  GdmaTsField rest_first_tensors_;
  for (uint32_t i = 0; i < first_tensor_timestep.size(); ++i) {
    move_valid = true;
    auto v = first_tensor_timestep[i];
    for (auto op : last_row_iter->tpu0_ts_field) {
      auto opds = op->getOperands();
      if (std::find(opds.begin(), opds.end(), v) != opds.end()) {
        move_valid = false;
        rest_first_tensors_.push_back(v);
        break;
      }
    }

    if (move_valid) {
      last_row_iter->gdma0_ts_field.push_back(v);
    }
  }

  if (!rest_first_tensors_.empty()) {
    TimestepRow new_row;
    new_row.gdma0_ts_field = rest_first_tensors_;
    timestep_table.push_back(new_row);
  }
  return stage_num;
}

/******************** The following is TimeStep *******************************/
BasicTimeStep::BasicTimeStep() {
  swpipl = std::make_shared<SoftwarePipeline>();
  this->clear();
}

void BasicTimeStep::clear() {
  timestep_table.clear();
  swpipl_stage_num = 1;
}

void BasicTimeStep::add_tpu0_ts_field(const TpuTsField &field) {
  TimestepRow row;
  row.tpu0_ts_field = field;
  timestep_table.push_back(row);
}

void BasicTimeStep::add_gdma0_ts_field(const GdmaTsField &field) {
  TimestepRow row;
  row.gdma0_ts_field = field;
  timestep_table.push_back(row);
}

void BasicTimeStep::add_tpu0_gdma0_ts_field(const TpuTsField &tpu_field,
                                            const GdmaTsField &gdma_field) {
  TimestepRow row;
  row.tpu0_ts_field = tpu_field;
  row.gdma0_ts_field = gdma_field;
  timestep_table.push_back(row);
}

int BasicTimeStep::get_layer_swpipl_stage(Operation *op) {
  return swpipl_stage_num == 1 ? 0 : 1;
}

int BasicTimeStep::get_tensor_swpipl_stage(Value v) {
  if (swpipl_stage_num == 1)
    return 0;

  return swpipl->get_tensor_swpipl_stage(v);
}

void BasicTimeStep::software_pipeline() {
  this->swpipl_stage_num =
      swpipl->software_pipeline_schedule(this->timestep_table);
}

void BasicTimeStep::show_timestep() {
  llvm::errs() << "====================================\n";
  llvm::errs() << "== show time step\n";
  std::string s;
  llvm::raw_string_ostream ss(s);
  for (int time_idx = 0; time_idx < this->get_timestep_num(); ++time_idx) {
    s.clear();
    ss << "=====Time step " << time_idx << "=====\n";
    const TpuTsField &layer_to_execute = timestep_table[time_idx].tpu0_ts_field;
    for (uint32_t i = 0; i < layer_to_execute.size(); ++i) {
      auto layer = layer_to_execute[i];
      ss << "==layer: ";
      layer->print(ss);
      ss << "(stage=" << this->get_layer_swpipl_stage(layer) << ")\n";
    }

    const GdmaTsField &tensor_load_store =
        timestep_table[time_idx].gdma0_ts_field;
    for (uint32_t i = 0; i < tensor_load_store.size(); ++i) {
      auto tensor = tensor_load_store[i];
      ss << "==tensor: ";
      tensor.print(ss);
      ss << "(stage=" << this->get_tensor_swpipl_stage(tensor) << ")\n";
    }
    ss << "\n";
    llvm::errs() << s;
  }
  llvm::errs() << "====================================\n";
}
