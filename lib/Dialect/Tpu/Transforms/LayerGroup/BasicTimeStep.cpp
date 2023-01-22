//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/BasicTimeStep.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace tpu {

using namespace mlir;
using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

BasicTimeStep::BasicTimeStep() {
  swpipl_ = std::make_shared<SoftwarePipeline>();
  timestep_method_ = std::make_shared<TimeStepMethod>();
  this->clear();
}

void BasicTimeStep::clear() {
  timestep_table_.clear();
  hold_coeff_.clear();
  lmem_buffer_.clear();
  lmem_occupy_ = 0;
  swpipl_stage_num_ = 1;
}

bool BasicTimeStep::assignTimeStep(const LgInfo &lg_info,
                                   const shape_secs_t &shape_secs,
                                   bool gen_idx) {
  clear();
  return timestep_method_->process(this, tensor_infos_, lg_info, shape_secs,
                                   gen_idx);
}

void BasicTimeStep::add_tpu0_ts_field(const TpuTsField &field) {
  TimestepRow row;
  row.tpu0_ts_field = field;
  timestep_table_.push_back(row);
}

void BasicTimeStep::add_gdma0_ts_field(const GdmaTsField &field) {
  TimestepRow row;
  row.gdma0_ts_field = field;
  timestep_table_.push_back(row);
}

void BasicTimeStep::add_tpu0_gdma0_ts_field(const TpuTsField &tpu_field,
                                            const GdmaTsField &gdma_field) {
  TimestepRow row;
  row.tpu0_ts_field = tpu_field;
  row.gdma0_ts_field = gdma_field;
  timestep_table_.push_back(row);
}

void BasicTimeStep::update_gdma0_ts_field(int64_t ts,
                                          const GdmaTsField &field) {
  this->timestep_table_[ts].gdma0_ts_field.clear();
  this->timestep_table_[ts].gdma0_ts_field = field;
}

int64_t BasicTimeStep::get_layer_swpipl_stage(Operation *op) {
  return swpipl_stage_num_ == 1 ? 0 : 1;
}

int64_t BasicTimeStep::get_tensor_swpipl_stage(Value v) {
  if (swpipl_stage_num_ == 1)
    return 0;

  return swpipl_->get_tensor_swpipl_stage(v);
}

void BasicTimeStep::software_pipeline() {
  this->swpipl_stage_num_ =
      swpipl_->software_pipeline_schedule(this->timestep_table_);
  for (auto &row : timestep_table_) {
    for (auto &iter : row.gdma0_ts_field) {
      iter.second.stage = get_tensor_swpipl_stage(iter.first);
    }
  }
}

void BasicTimeStep::show_timestep() {
  llvm::errs() << "============= show time step =============\n";
  size_t timestep_num = get_timestep_num();
  std::string s;
  llvm::raw_string_ostream ss(s);

  ValueIntMap value_ids;
  std::map<Operation *, int64_t> op_ids;
  int64_t idx = 0;
  for (size_t ts = 0; ts < timestep_num; ++ts) {
    auto &layer_field = getLayers(ts);
    for (auto op : layer_field) {
      if (op_ids.find(op) == op_ids.end()) {
        op_ids[op] = idx++;
      }

      for (auto in : op->getOperands()) {
        if (value_ids.find(in) == value_ids.end()) {
          value_ids[in] = idx++;
        }
      }
      for (auto out : get_output_values(op)) {
        if (value_ids.find(out) == value_ids.end()) {
          value_ids[out] = idx++;
        }
      }
    }
  }

  mem_buffer_key_t buffer_key;
  for (size_t ts = 0; ts < timestep_num; ++ts) {
    s.clear();
    ss << "=== timestep " << ts << ": \n";
    const auto &layer_field = getLayers(ts);
    for (auto op : layer_field) {
      ss << "layer " << op_ids[op] << "([";
      for (auto in : op->getOperands()) {
        if (in.getType().isa<NoneType>()) {
          continue;
        }
        buffer_key.value = in;
        if (dyn_cast_or_null<top::WeightOp>(in.getDefiningOp())) {
          buffer_key.type = LMEM_WEIGHT;
        } else {
          buffer_key.type = LMEM_ACTIVATION;
        }
        auto &buffer_value = get_lmem_buffer_value(buffer_key);
        ss << value_ids[in] << "(" << buffer_value.start_ts << ", "
           << buffer_value.end_ts << "), ";
      }
      ss << "] -> [";
      for (auto out : get_output_values(op)) {
        buffer_key.type = LMEM_ACTIVATION;
        buffer_key.value = out;
        auto &buffer_value = get_lmem_buffer_value(buffer_key);
        ss << value_ids[out] << "(" << buffer_value.start_ts << ", "
           << buffer_value.end_ts << "), ";
      }
      ss << "])\n";
    }

    const auto &tensor_field = getTensors(ts);
    ss << "tensor(start_ts, end_ts): ";
    for (auto &iter : tensor_field) {
      buffer_key.value = iter.first;
      if (dyn_cast_or_null<top::WeightOp>(iter.first.getDefiningOp())) {
        buffer_key.type = LMEM_WEIGHT;
      } else {
        buffer_key.type = LMEM_ACTIVATION;
      }
      auto &buffer_value = get_lmem_buffer_value(buffer_key);
      ss << value_ids[iter.first] << "(" << buffer_value.start_ts << ", "
         << buffer_value.end_ts << "), ";
    }
    ss << "\n";
    llvm::errs() << s;
  }
  llvm::errs() << "====================================\n";
}

void BasicTimeStep::gen_hold_coeff() {
  this->hold_coeff_.clear();

  Value v;
  for (size_t ts = 0; ts < this->get_timestep_num(); ++ts) {
    const GdmaTsField &gdma_field = this->timestep_table_[ts].gdma0_ts_field;
    for (size_t i = 0; i < gdma_field.size(); ++i) {
      v = gdma_field[i].first;
      auto &tensor_info = gdma_field[i].second;
      if (tensor_info.mode == TIMESTEP_LOAD) {
        if (module::isWeight(v)) {
          this->hold_coeff_[v] = ts;
        }
      }
    }
  }
}

// int64_t BasicTimeStep::get_start_timestep(Value v) {
//   for (auto &lmem_info : lmem_buffer_) {
//     if (lmem_info.first.value == v) {
//       return lmem_info.second.start_ts;
//     }
//   }
// }

void BasicTimeStep::gen_all_mem_buffer() {
  // input: need_imm_buffers
  lmem_buffer_.clear();

  mem_buffer_key_t lmem_key;
  mem_buffer_value_t lmem_value = {0};
  lmem_value.align_bytes = 32;

  for (int64_t stg = 0; stg < this->swpipl_stage_num_; ++stg) {
    // add for software pipeline
    bool layer_timestep_valid =
        (swpipl_stage_num_ == 1) || (swpipl_stage_num_ > 1 && stg == 1);
    for (size_t ts = 0; ts < get_timestep_num(); ++ts) {
      // process current timestep layers
      const TpuTsField &cur_tpu_field = timestep_table_[ts].tpu0_ts_field;
      if (layer_timestep_valid) {
        for (auto op : cur_tpu_field) {
          // Results
          for (auto out : get_output_values(op)) {
            // Need some process for concat opt case
            lmem_key.value = out;
            lmem_key.type = LMEM_ACTIVATION;

            lmem_value.start_ts = ts;
            lmem_value.end_ts = -1;

            lmem_buffer_[lmem_key] = lmem_value;
          }

          // Operands
          for (auto in : op->getOperands()) {
            if (in.getType().isa<NoneType>()) {
              continue;
            }
            if (module::isWeight(in)) {
              lmem_key.type = LMEM_WEIGHT;
            } else {
              lmem_key.type = LMEM_ACTIVATION;
            }
            lmem_key.value = in;

            lmem_buffer_[lmem_key].end_ts = ts;
          }

          // imm buffer
          lmem_key.op = op;
          lmem_key.type = LMEM_OPERATION;

          lmem_value.start_ts = ts;
          lmem_value.end_ts = ts;

          lmem_buffer_[lmem_key] = lmem_value;
        } // cur_tpu_field
      }
      // process current timestep tensors
      const GdmaTsField &cur_gdma_field = timestep_table_[ts].gdma0_ts_field;
      for (auto &tensor : cur_gdma_field) {
        auto &tensor_info = tensor.second;
        if (tensor_info.stage != stg) {
          continue;
        }

        if (tensor_info.mode == TIMESTEP_LOAD) {
          // Need some process for concat opt case
          if (module::isWeight(tensor.first)) {
            lmem_key.type = LMEM_WEIGHT;
          } else {
            lmem_key.type = LMEM_ACTIVATION;
          }
          lmem_key.value = tensor.first;

          lmem_value.start_ts = ts;
          lmem_value.end_ts = -1;

          lmem_buffer_[lmem_key] = lmem_value;
        } else if (tensor_info.mode == TIMESTEP_STORE) {
          lmem_key.value = tensor.first;
          lmem_key.type = LMEM_ACTIVATION;

          lmem_buffer_[lmem_key].end_ts = ts;
        }
      }
    }
  }
}

void BasicTimeStep::update_all_mem_buffer_size(const LgInfo &lg_info) {
  if (lmem_buffer_.empty()) {
    gen_all_mem_buffer();
  }
  auto &tensor_infos = tensor_infos_;

  int64_t nslice, hslice;
  for (auto iter = lmem_buffer_.begin(); iter != lmem_buffer_.end(); ++iter) {
    if (iter->first.type == LMEM_OPERATION) {
      continue;
    }
    auto v = iter->first.value;
    auto &tensor_info = tensor_infos[v];
    auto &si = tensor_info.slice_info;
    if (iter->first.type == LMEM_ACTIVATION) {
      get_max_slice_nh(si, nslice, hslice);
      iter->second.size =
          Arch::get_tensor_lmem_bytes(v, nslice, hslice, tensor_info.eu_align);
    } else if (iter->first.type == LMEM_WEIGHT) {
      iter->second.size = Arch::get_weight_lmem_bytes(v, tensor_info.eu_align);
    }
  }

  mem_buffer_key_t buffer_key0;
  mem_buffer_key_t buffer_key1;
  buffer_key0.type = LMEM_ACTIVATION;
  buffer_key1.type = LMEM_ACTIVATION;
  int64_t in_nslice, in_hslice, out_nslice, out_hslice;
  for (auto iter = lmem_buffer_.begin(); iter != lmem_buffer_.end(); ++iter) {
    if (iter->first.type == LMEM_OPERATION) {
      auto op = iter->first.op;
      auto in = op->getOperand(0);
      auto out = op->getResult(0);
      auto &in_ti = tensor_infos[in];
      auto &out_ti = tensor_infos[out];

      get_max_slice_nh(in_ti.slice_info, in_nslice, in_hslice);
      get_max_slice_nh(out_ti.slice_info, out_nslice, out_hslice);
      buffer_key0.value = in;
      buffer_key1.value = out;

      auto lg_op = cast<LocalGenInterface>(op);
      iter->second.size = lg_op.getBufferSize(
          lmem_buffer_[buffer_key0].size, lmem_buffer_[buffer_key1].size,
          in_nslice, in_hslice, out_nslice, out_hslice);
    }
  }

  // erase mem_buffer whose size=0 because these ops don't need imm_buffer
  std::map<mem_buffer_key_t, mem_buffer_value_t>::iterator iter;
  for (iter = lmem_buffer_.begin(); iter != lmem_buffer_.end();) {
    iter = iter->second.size == 0 ? lmem_buffer_.erase(iter) : std::next(iter);
  }
}

const mem_buffer_value_t &
BasicTimeStep::get_lmem_buffer_value(const mem_buffer_key_t &buffer_key) {
  //
  return lmem_buffer_[buffer_key];
}

void BasicTimeStep::set_lmem_addr(const mem_buffer_key_t &buffer_key,
                                  int64_t lmem_addr) {
  lmem_buffer_[buffer_key].addr = lmem_addr;
}

int64_t BasicTimeStep::get_lmem_addr(const mem_buffer_key_t &buffer_key) {
  return lmem_buffer_[buffer_key].addr;
}
int64_t BasicTimeStep::get_lmem_size(const mem_buffer_key_t &buffer_key) {
  return lmem_buffer_[buffer_key].size;
}

bool BasicTimeStep::is_tensor_hold_in_lmem(Value v) {
  std::map<Value, int64_t, value_compare>::iterator iter =
      this->hold_coeff_.find(v);
  return (iter != this->hold_coeff_.end());
}

TensorInfo &BasicTimeStep::get_tensor_infos() { return tensor_infos_; }

typedef struct {
  Value value;
  int64_t addr;
  int64_t size;
} eltwise_to_show_t;

void BasicTimeStep::show_lmem_buffer() {
  llvm::errs() << "====================================\n";
  llvm::errs() << "== show lmem buffer\n";
  mem_buffer_key_t key;
  mem_buffer_value_t value;
  size_t timestep_num = get_timestep_num();
  std::vector<std::vector<std::pair<int64_t, int64_t>>> data(
      timestep_num, std::vector<std::pair<int64_t, int64_t>>());
  for (auto &iter : lmem_buffer_) {
    key = iter.first;
    value = iter.second;
    bool first_step = true;
    for (int64_t ts = value.start_ts;
         ts != ((value.end_ts + 1) % timestep_num) || first_step;
         ts = (ts + 1) % timestep_num) {
      first_step = false;
      data[ts].push_back(std::make_pair(value.addr, value.size));
    }
  }

  for (size_t ts = 0; ts < timestep_num; ++ts) {
    llvm::errs() << "=== timestep = " << ts << "\n";
    llvm::errs() << "addr(end): ";
    int64_t total = 0;
    std::sort(data[ts].begin(), data[ts].end());
    for (auto &iter : data[ts]) {
      total += iter.second;
      llvm::errs() << "(" << iter.first << ", " << iter.first + iter.second
                   << "), ";
    }
    llvm::errs() << "total=" << total << "\n";
  }

  llvm::errs() << "====================================\n";
}

int64_t BasicTimeStep::get_tensor_range_end(const GdmaElt &tensor,
                                            int64_t cur_ts) {
  int64_t timestep_num = this->get_timestep_num();

  assert(this->swpipl_stage_num_ == 3);
  TIMESTEP_LD_ST gdma_type = tensor.second.mode;

  int64_t result = 0;
  bool find_flag = false;
  auto v = tensor.first;
  if (is_timestep_load(gdma_type)) {
    result = 0;
    for (int64_t ts = cur_ts; ts >= 0; --ts) {
      // layers
      auto &ts_layers = timestep_table_[ts].tpu0_ts_field;
      for (auto op : ts_layers) {
        auto ins = op->getOperands();
        find_flag = std::find(ins.begin(), ins.end(), v) != ins.end();
        if (find_flag) {
          result = ts + 1;
          break;
        }
      }
      if (find_flag) {
        break;
      }
    }
  } else {
    result = timestep_num - 1;
    for (int64_t ts = cur_ts; ts < timestep_num; ++ts) {
      // layers
      auto &ts_layers = timestep_table_[ts].tpu0_ts_field;
      for (auto op : ts_layers) {
        auto outs = op->getResults();
        find_flag = std::find(outs.begin(), outs.end(), v) != outs.end();
        if (find_flag) {
          result = std::min(result, ts - 1);
          break;
        }
      }
      if (find_flag) {
        break;
      }
    }
  }
  return result;
}

int64_t BasicTimeStep::get_tensor_life_time(Value v) {
  mem_buffer_key_t key = {LMEM_ACTIVATION, v, nullptr};
  if (module::isWeight(v)) {
    key.type = LMEM_WEIGHT;
  }
  auto &buffer_value = this->get_lmem_buffer_value(key);
  int64_t life_time = 0;
  int64_t timestep_num = this->get_timestep_num();
  int64_t start_ts = buffer_value.start_ts;
  int64_t end_ts = buffer_value.end_ts;
  if (end_ts >= start_ts) {
    life_time = end_ts - start_ts + 1;
  } else {
    life_time = timestep_num - start_ts + end_ts + 1;
  }
  return life_time;
}

void BasicTimeStep::cancel_tensor_hold_in_lmem(Value v) {
  auto iter = this->hold_coeff_.find(v);
  this->hold_coeff_.erase(iter);
}

} // namespace tpu
} // namespace tpu_mlir
