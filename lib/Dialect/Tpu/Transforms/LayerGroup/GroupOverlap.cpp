//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOverlap.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace tpu {

static bool used_buffer_cmp(const MemBlock &lhs, const MemBlock &rhs) {
  return lhs.first < rhs.first;
}

void find_used_banks(std::set<int64_t> &used_banks, int64_t lmem_addr,
                     int64_t lmem_size) {
  int64_t bank_size = Arch::LMEM_BANK_BYTES;
  int64_t start_bank = lmem_addr / bank_size;
  int64_t end_bank = (lmem_addr + lmem_size - 1) / bank_size;
  for (int64_t i = start_bank; i <= end_bank; ++i) {
    used_banks.insert(i);
  }
}

// return the earliest timestep idx of the up layer group
static std::vector<int64_t>
up_overlap_depth(BasicTimeStepPtr &up_time_step,
                 const std::vector<MemBlock> &overlap_buffer,
                 const std::vector<Value> &overlap_tensor,
                 const LgInfo &up_group, const shape_secs_t &up_shape_secs) {
  int64_t up_ts_num = up_time_step->get_timestep_num();
  std::vector<int64_t> up_timestep_depth(overlap_buffer.size(), up_ts_num);
  // If the loop number of the up group is <= 2 and the software
  // pipeline is opened and the tensor is the output of up_group,
  // we can't put the down ops to the up group.
  // Because data that are loaded in the next group may be not
  // stored in the up group.
  // auto &up_group_outs = up_group.group_outs;
  if ((up_shape_secs.nsecs * up_shape_secs.hsecs <= 2) &&
      (up_time_step->get_swpipl_stage_num() > 1)) {
    return up_timestep_depth;
  }

  // up group used buffer
  auto &up_mem_buffer = up_time_step->get_lmem_buffer();
  std::vector<std::list<MemBlock>> up_used_buffer(up_ts_num);
  for (auto iter = up_mem_buffer.begin(); iter != up_mem_buffer.end(); ++iter) {
    int64_t start_ts = iter->second.start_ts;
    int64_t end_ts = iter->second.end_ts;
    if (start_ts <= end_ts) {
      for (int64_t ts = start_ts; ts <= end_ts; ++ts) {
        up_used_buffer[ts].push_back(
            std::make_pair(iter->second.addr, iter->second.size));
      }
    } else {
      // whether to consider the mem buffer or not
      Value tensor = iter->first.value;
      auto &up_ts_tensors = up_time_step->getTensors(start_ts);
      bool consider = true;
      for (size_t i = 0; i < up_ts_tensors.size(); ++i) {
        if (up_ts_tensors[i].second.mode == TIMESTEP_LOAD &&
            up_ts_tensors[i].first == tensor) {
          if (up_ts_tensors[i].second.stage < 1) {
            consider = false;
          }
        }
      }
      for (int64_t j = 0; j <= end_ts; ++j) {
        up_used_buffer[j].push_back(
            std::make_pair(iter->second.addr, iter->second.size));
      }
      if (consider) {
        for (int64_t j = start_ts; j < up_ts_num; ++j) {
          up_used_buffer[j].push_back(
              std::make_pair(iter->second.addr, iter->second.size));
        }
      }
    }
  }
  // sort up_used_buffer
  for (size_t i = 0; i < up_used_buffer.size(); ++i) {
    up_used_buffer[i].sort(used_buffer_cmp);
  }
  // get up timestep depth for overlap buffer
  for (size_t i = 0; i < overlap_buffer.size(); ++i) {
    int64_t up_depth = up_ts_num;
    int64_t overlap_buffer_addr = overlap_buffer[i].first;
    int64_t overlap_buffer_size = overlap_buffer[i].second;
    for (int64_t j = up_ts_num - 1; j >= 0; --j) {
      bool conflict = false;
      auto &up_ts_tensors = up_time_step->getTensors(j);
      for (size_t k = 0; k < up_ts_tensors.size(); ++k) {
        if (up_ts_tensors[k].second.mode == TIMESTEP_STORE &&
            up_ts_tensors[k].first == overlap_tensor[i]) {
          conflict = true;
          break;
        }
      }
      if (!conflict) {
        for (auto list_it = up_used_buffer[j].begin();
             list_it != up_used_buffer[j].end(); ++list_it) {
          int64_t used_addr = list_it->first;
          int64_t used_size = list_it->second;
          if (!(used_addr >= overlap_buffer_addr + overlap_buffer_size ||
                overlap_buffer_addr >= used_addr + used_size)) {
            conflict = true;
            break;
          } else if (used_addr >= overlap_buffer_addr + overlap_buffer_size) {
            break;
          }
        }
      }
      if (!conflict) {
        up_depth = j;
      } else {
        break;
      }
    }
    up_timestep_depth[i] = up_depth;
  }
  return up_timestep_depth;
}

// return the timestep idx of the up layer group
static std::vector<int64_t>
down_overlap_depth(BasicTimeStepPtr &down_time_step,
                   const std::vector<MemBlock> &overlap_buffer,
                   const std::vector<Value> &overlap_tensor) {
  int64_t down_ts_num = down_time_step->get_timestep_num();
  std::vector<int64_t> down_timestep_depth(overlap_buffer.size(), down_ts_num);

  // down group used buffer
  auto &down_mem_buffer = down_time_step->get_lmem_buffer();
  std::vector<std::list<MemBlock>> down_used_buffer(down_ts_num);
  for (auto iter = down_mem_buffer.begin(); iter != down_mem_buffer.end();
       ++iter) {
    int64_t start_ts = iter->second.start_ts;
    int64_t end_ts = iter->second.end_ts;
    if (start_ts <= end_ts) {
      for (int64_t ts = start_ts; ts <= end_ts; ++ts) {
        down_used_buffer[ts].push_back(
            std::make_pair(iter->second.addr, iter->second.size));
      }
    } else {
      // whether to consider the mem bufer or not
      Value tensor = iter->first.value;
      auto &down_ts_tensors = down_time_step->getTensors(start_ts);
      bool consider = false;
      for (size_t i = 0; i < down_ts_tensors.size(); ++i) {
        if (down_ts_tensors[i].second.mode == TIMESTEP_LOAD &&
            down_ts_tensors[i].first == tensor) {
          if (down_ts_tensors[i].second.stage < 1) {
            consider = false;
          }
        }
      }
      for (int64_t j = 0; j <= end_ts; ++j) {
        down_used_buffer[j].push_back(
            std::make_pair(iter->second.addr, iter->second.size));
      }
      if (consider) {
        for (int64_t j = start_ts; j < down_ts_num; ++j) {
          down_used_buffer[j].push_back(
              std::make_pair(iter->second.addr, iter->second.size));
        }
      }
    }
  }
  // sort down_used_buffer
  for (size_t i = 0; i < down_used_buffer.size(); ++i) {
    down_used_buffer[i].sort(used_buffer_cmp);
  }
  // get down timestep depth for overlap buffer
  for (size_t i = 0; i < overlap_buffer.size(); ++i) {
    int64_t down_depth = -1;

    int64_t overlap_buffer_addr = overlap_buffer[i].first;
    int64_t overlap_buffer_size = overlap_buffer[i].second;
    for (int64_t j = 0; j < down_ts_num; ++j) {
      bool conflict = false;
      auto &down_ts_tensors = down_time_step->getTensors(j);
      for (size_t k = 0; k < down_ts_tensors.size(); ++k) {
        if (down_ts_tensors[k].second.mode == TIMESTEP_LOAD &&
            down_ts_tensors[k].first == overlap_tensor[i]) {
          conflict = true;
          break;
        }
      }
      if (!conflict) {
        for (auto list_it = down_used_buffer[j].begin();
             list_it != down_used_buffer[j].end(); ++list_it) {
          int64_t used_addr = list_it->first;
          int64_t used_size = list_it->second;
          if (!(used_addr >= overlap_buffer_addr + overlap_buffer_size ||
                overlap_buffer_addr >= used_addr + used_size)) {
            conflict = true;
            break;
          } else if (used_addr >= overlap_buffer_addr + overlap_buffer_size) {
            break;
          }
        }
      }
      if (!conflict) {
        down_depth = j;
      } else {
        break;
      }
    }
    down_timestep_depth[i] = down_depth;
  }
  return down_timestep_depth;
}

static bool buffer_conflict_with_layer(BasicTimeStepPtr &cur_time_step,
                                       int64_t cur_ts,
                                       const MemBlock &buffer_locate) {
  MemBlock lmem_locate;
  std::set<int64_t> layer_used_banks;
  auto ts_layers = cur_time_step->getLayers(cur_ts);
  for (size_t i = 0; i < ts_layers.size(); ++i) {
    auto op = ts_layers[i];
    auto ins = get_input_values(op);
    for (auto in : ins) {
      lmem_locate = cur_time_step->get_lmem_locate(in, cur_ts);
      if (lmem_locate.first < 0) {
        continue;
      }
      find_used_banks(layer_used_banks, lmem_locate.first, lmem_locate.second);
    }
    auto outs = get_output_values(op);
    for (auto out : outs) {
      lmem_locate = cur_time_step->get_lmem_locate(out, cur_ts);
      if (lmem_locate.first < 0) {
        continue;
      }
      find_used_banks(layer_used_banks, lmem_locate.first, lmem_locate.second);
    }
    mem_buffer_key_t buffer_key;
    buffer_key.type = LMEM_OPERATION;
    buffer_key.op = op;
    auto &cur_lmem_buffer = cur_time_step->get_lmem_buffer();
    auto iter = cur_lmem_buffer.find(buffer_key);
    if (iter != cur_lmem_buffer.end()) {
      lmem_locate.first = iter->second.addr;
      lmem_locate.second = iter->second.size;
      find_used_banks(layer_used_banks, iter->second.addr, iter->second.size);
    }
  }
  std::set<int64_t> buffer_used_banks;
  find_used_banks(buffer_used_banks, buffer_locate.first, buffer_locate.second);
  bool conflict = false;
  for (auto iter = buffer_used_banks.begin(); iter != buffer_used_banks.end();
       ++iter) {
    if (layer_used_banks.find(*iter) != layer_used_banks.end()) {
      conflict = true;
      break;
    }
  }
  return conflict;
}

static void assign_down_to_up_overlap_timestep(
    // up group
    const LgInfo &up_group, const shape_secs_t &up_secs,
    BasicTimeStepPtr &up_time_step,
    // down group
    const LgInfo &down_group, const shape_secs_t &down_secs,
    BasicTimeStepPtr &down_time_step,
    std::vector<std::pair<Value, int64_t>> &down_group_overlap_op,
    std::vector<int64_t> &down_to_up_depth,
    std::vector<MemBlock> &down_to_up_overlap_buffer) {
  std::shared_ptr<CycleCalculator> cycle_calculator;
  if (module::isCV18xx()) {
    cycle_calculator = std::make_shared<Cv18xxCycleCalculator>();
  } else {
    cycle_calculator = std::make_shared<Bm168xCycleCalculator>();
  }
  // get down group overlap gdma cycle count
  int64_t timestep_idx, idx;
  std::vector<int64_t> overlap_gdma_cycle;
  auto &down_tensor_infos = down_time_step->get_tensor_infos();
  for (size_t i = 0; i < down_group_overlap_op.size(); ++i) {
    Value tensor = down_group_overlap_op[i].first;
    timestep_idx = down_group_overlap_op[i].second;
    auto &ti = down_tensor_infos[tensor];
    auto &ts_tensors = down_time_step->getTensors(timestep_idx);
    for (idx = 0; idx < ts_tensors.size(); ++idx) {
      if (ts_tensors[idx].first == tensor) {
        break;
      }
    }
    if (idx == ts_tensors.size()) {
      llvm_unreachable("cannot find down group tensor");
    }
    int64_t cycle_count =
        cycle_calculator->getGdmaCycle(tensor, ti, down_group.type);
    overlap_gdma_cycle.push_back(cycle_count);
  }

  // assignment
  std::map<int64_t, int64_t> timestep_slack;
  int64_t up_ts_num = up_time_step->get_timestep_num();
  int64_t up_swpipl_stage_num = up_time_step->get_swpipl_stage_num();
  int64_t least_up_stage = up_swpipl_stage_num > 1 ? 1 : 0;
  std::vector<int64_t> sel_up_overlap_ts;
  auto &up_tensor_infos = up_time_step->get_tensor_infos();
  for (size_t i = 0; i < down_to_up_depth.size(); ++i) {
    int64_t cur_sel_overlap_ts = -1;
    int64_t max_profit = -1;
    int64_t cur_profit = -1;
    int64_t min_remain_slack = -1;
    int64_t cur_remain_slack = -1;
    bool conflict_with_layer = false;
    bool cur_conflict_with_layer;
    for (int64_t ts = up_ts_num - 1; ts >= down_to_up_depth[i]; --ts) {
      // get timestep slack
      if (timestep_slack.find(ts) == timestep_slack.end()) {
        timestep_slack[ts] = 0;
        auto &ts_layers = up_time_step->getLayers(ts);
        auto &ts_tensors = up_time_step->getTensors(ts);
        for (size_t j = 0; j < ts_layers.size(); ++j) {
          timestep_slack[ts] += cycle_calculator->getLocalLayerCycle(
              ts_layers[j], up_tensor_infos, up_group.type, true);
        }
        for (size_t j = 0; j < ts_tensors.size(); ++j) {
          if (ts_tensors[j].second.stage >= least_up_stage) {
            timestep_slack[ts] -= cycle_calculator->getGdmaCycle(
                ts_tensors[j].first, up_tensor_infos[ts_tensors[j].first],
                up_group.type);
          }
        }
      }
      // look for the up overlap timestep
      int64_t cur_slack = timestep_slack[ts];
      if (cur_slack > 0) {
        cur_conflict_with_layer = buffer_conflict_with_layer(
            up_time_step, ts, down_to_up_overlap_buffer[i]);
        cur_profit = std::min(overlap_gdma_cycle[i], cur_slack);
        cur_remain_slack = cur_slack - overlap_gdma_cycle[i];
        bool change = false;
        if (cur_profit > max_profit) {
          change = true;
        } else if (cur_profit == max_profit) {
          if (!cur_conflict_with_layer && conflict_with_layer) {
            change = true;
          } else if (cur_conflict_with_layer == conflict_with_layer) {
            if (cur_remain_slack < min_remain_slack) {
              change = true;
            }
          }
        }
        if (change) {
          cur_sel_overlap_ts = ts;
          max_profit = cur_profit;
          min_remain_slack = cur_remain_slack;
          conflict_with_layer = cur_conflict_with_layer;
        }
      }
    }
    sel_up_overlap_ts.push_back(cur_sel_overlap_ts);
    if (cur_sel_overlap_ts != -1) {
      timestep_slack[cur_sel_overlap_ts] -= overlap_gdma_cycle[i];
    }
  }

  // set timestep overlap info
  // get down group slice info
  for (size_t i = 0; i < down_group_overlap_op.size(); ++i) {
    if (sel_up_overlap_ts[i] >= 0) {
      Value tensor = down_group_overlap_op[i].first;
      down_time_step->insert_self_up_op(tensor);

      up_time_step->insert_other_up_op(tensor, sel_up_overlap_ts[i]);
      // auto &up_ts_tensors = up_time_step->getTensors(sel_up_overlap_ts[i]);
      // GdmaElt new_elt = std::make_pair(tensor, down_tensor_infos[tensor]);
      // up_ts_tensors.push_back(new_elt);
    }
  }
}

static void assign_up_to_down_overlap_timestep(
    // up group
    const LgInfo &up_group, const shape_secs_t &up_secs,
    BasicTimeStepPtr &up_time_step,
    // down group
    const LgInfo &down_group, const shape_secs_t &down_secs,
    BasicTimeStepPtr &down_time_step,
    std::vector<std::pair<Value, int64_t>> &up_group_overlap_op,
    std::vector<int64_t> &up_to_down_depth,
    std::vector<MemBlock> &up_to_down_overlap_buffer) {
  std::shared_ptr<CycleCalculator> cycle_calculator;
  if (module::isCV18xx()) {
    cycle_calculator = std::make_shared<Cv18xxCycleCalculator>();
  } else {
    cycle_calculator = std::make_shared<Bm168xCycleCalculator>();
  }
  // get up group overlap gdma cycle count
  int64_t timestep_idx, idx;
  std::vector<int64_t> overlap_gdma_cycle;
  auto &up_tensor_infos = up_time_step->get_tensor_infos();
  for (size_t i = 0; i < up_group_overlap_op.size(); ++i) {
    Value tensor = up_group_overlap_op[i].first;
    timestep_idx = up_group_overlap_op[i].second;
    auto &ti = up_tensor_infos[tensor];
    auto &ts_tensors = up_time_step->getTensors(timestep_idx);
    for (idx = 0; idx < ts_tensors.size(); ++idx) {
      if (ts_tensors[idx].first == tensor) {
        break;
      }
    }
    if (idx == ts_tensors.size()) {
      llvm_unreachable("cannot find down group tensor");
    }
    int64_t cycle_count =
        cycle_calculator->getGdmaCycle(tensor, ti, up_group.type);
    overlap_gdma_cycle.push_back(cycle_count);
  }

  // assignment
  std::map<int64_t, int64_t> timestep_slack;
  // int64_t down_ts_num = down_time_step->get_timestep_num();
  int64_t down_swpipl_stage_num = down_time_step->get_swpipl_stage_num();
  int64_t largest_down_stage = down_swpipl_stage_num > 1 ? 1 : 0;
  std::vector<int64_t> sel_down_overlap_ts;
  auto &down_tensor_infos = down_time_step->get_tensor_infos();
  for (size_t i = 0; i < up_to_down_depth.size(); ++i) {
    int64_t cur_sel_overlap_ts = -1;
    int64_t max_profit = -1;
    int64_t cur_profit = -1;
    int64_t min_remain_slack = -1;
    int64_t cur_remain_slack = -1;
    bool conflict_with_layer = false;
    bool cur_conflict_with_layer;
    for (int64_t ts = 0; ts <= up_to_down_depth[i]; ++ts) {
      // get timestep slack
      if (timestep_slack.find(ts) == timestep_slack.end()) {
        timestep_slack[ts] = 0;
        auto &ts_layers = down_time_step->getLayers(ts);
        auto &ts_tensors = down_time_step->getTensors(ts);
        for (size_t j = 0; j < ts_layers.size(); ++j) {
          timestep_slack[ts] += cycle_calculator->getLocalLayerCycle(
              ts_layers[j], down_tensor_infos, down_group.type, true);
        }
        for (size_t j = 0; j < ts_tensors.size(); ++j) {
          if (ts_tensors[j].second.stage <= largest_down_stage) {
            timestep_slack[ts] -= cycle_calculator->getGdmaCycle(
                ts_tensors[j].first, down_tensor_infos[ts_tensors[j].first],
                down_group.type);
          }
        }
      }
      // look for the up overlap timestep
      int64_t cur_slack = timestep_slack[ts];
      if (cur_slack > 0) {
        cur_conflict_with_layer = buffer_conflict_with_layer(
            down_time_step, ts, up_to_down_overlap_buffer[i]);
        cur_profit = std::min(overlap_gdma_cycle[i], cur_slack);
        cur_remain_slack = cur_slack - overlap_gdma_cycle[i];
        bool change = false;
        if (cur_profit > max_profit) {
          change = true;
        } else if (cur_profit == max_profit) {
          if (!cur_conflict_with_layer && conflict_with_layer) {
            change = true;
          } else if (cur_conflict_with_layer == conflict_with_layer) {
            if (cur_remain_slack < min_remain_slack) {
              change = true;
            }
          }
        }
        if (change) {
          cur_sel_overlap_ts = ts;
          max_profit = cur_profit;
          min_remain_slack = cur_remain_slack;
          conflict_with_layer = cur_conflict_with_layer;
        }
      }
    }
    sel_down_overlap_ts.push_back(cur_sel_overlap_ts);
    if (cur_sel_overlap_ts != -1) {
      timestep_slack[cur_sel_overlap_ts] -= overlap_gdma_cycle[i];
    }
  }

  // set timestep overlap info
  // get up group slice info
  for (size_t i = 0; i < up_group_overlap_op.size(); ++i) {
    if (sel_down_overlap_ts[i] >= 0) {
      Value tensor = up_group_overlap_op[i].first;
      up_time_step->insert_self_down_op(tensor);

      down_time_step->insert_other_down_op(tensor, sel_down_overlap_ts[i]);
      // auto &down_ts_tensors =
      // down_time_step->getTensors(sel_down_overlap_ts[i]);
      // GdmaElt new_elt = std::make_pair(tensor, up_tensor_infos[tensor]);
      // down_ts_tensors.push_back(new_elt);
    }
  }
}

static bool connect_neuron_overlap_valid(Value tensor,
                                         const BasicTimeStepPtr &up_time_step,
                                         const shape_secs_t &up_secs,
                                         const BasicTimeStepPtr &down_time_step,
                                         const shape_secs_t &down_secs,
                                         bool dynamic_compile) {
  bool valid = false;
  if (dynamic_compile) {
    valid = false;
  } else if (up_secs.nsecs > 1 && down_secs.nsecs > 1) {
    valid = true;
  } else if (up_secs.nsecs == 1 && down_secs.nsecs != 1) {
    valid = false;
  } else if (up_secs.nsecs != 1 && down_secs.nsecs == 1) {
    valid = false;
  } else if ((up_secs.hsecs > 1 && down_secs.hsecs > 1)) {
    assert(up_secs.nsecs == 1 && down_secs.nsecs == 1);

    auto &up_tensor_infos = up_time_step->get_tensor_infos();
    auto &up_si = up_tensor_infos[tensor].slice_info;
    int64_t up_hidx = up_si.h[up_secs.hsecs - 1].first;

    auto &down_tensor_infos = down_time_step->get_tensor_infos();
    auto &down_si = down_tensor_infos[tensor].slice_info;
    int64_t down_hslice = down_si.h[0].first + down_si.h[0].second;

    valid = (down_hslice <= up_hidx);
  }
  return valid;
}

static void find_alter_overlap_op(
    // up group
    const LgInfo &up_group, const BasicTimeStepPtr &up_time_step,
    const shape_secs_t up_secs,
    std::vector<std::pair<Value, int64_t>> &up_group_overlap_op,
    // down group
    const LgInfo &down_group, const BasicTimeStepPtr &down_time_step,
    const shape_secs_t down_secs,
    std::vector<std::pair<Value, int64_t>> &down_group_overlap_op,
    bool dynamic_compile) {
  Value tensor;
  bool overlap_valid = false;
  // gdma op of down group overlap to up group
  auto &up_group_outs = up_group.group_outs;
  assert(down_time_step->get_swpipl_stage_num() == 3);
  int64_t down_ts_num = down_time_step->get_timestep_num();
  for (int64_t ts = 0; ts < down_ts_num; ++ts) {
    auto &ts_tensors = down_time_step->getTensors(ts);
    for (size_t i = 0; i < ts_tensors.size(); ++i) {
      tensor = ts_tensors[i].first;
      overlap_valid = false;
      if (ts_tensors[i].second.stage == 0 &&
          ts_tensors[i].second.mode == TIMESTEP_LOAD) {
        overlap_valid = true;
        if (module::isWeight(tensor) == false &&
            std::find(up_group_outs.begin(), up_group_outs.end(), tensor) !=
                up_group_outs.end()) {
          overlap_valid = connect_neuron_overlap_valid(
              tensor, up_time_step, up_secs, down_time_step, down_secs,
              dynamic_compile);
        }
      }
      if (overlap_valid) {
        down_group_overlap_op.push_back(std::make_pair(tensor, ts));
      }
    }
  }

  // gdma op of up group overlap to down group
  assert(up_time_step->get_swpipl_stage_num() == 3);
  auto &down_group_ins = down_group.group_ins;
  int64_t up_ts_num = up_time_step->get_timestep_num();
  for (int64_t ts = 0; ts < up_ts_num; ++ts) {
    auto &ts_tensors = up_time_step->getTensors(ts);
    for (size_t i = 0; i < ts_tensors.size(); ++i) {
      tensor = ts_tensors[i].first;
      overlap_valid = false;
      if (ts_tensors[i].second.stage == 2 &&
          ts_tensors[i].second.mode == TIMESTEP_STORE) {
        overlap_valid = true;
        if (std::find(down_group_ins.begin(), down_group_ins.end(), tensor) !=
            down_group_ins.end()) {
          overlap_valid = connect_neuron_overlap_valid(
              tensor, up_time_step, up_secs, down_time_step, down_secs,
              dynamic_compile);
        }
      }
      if (overlap_valid) {
        up_group_overlap_op.push_back(std::make_pair(tensor, ts));
      }
    }
  }
}

static void
direct_group_overlap_schd(std::vector<BasicTimeStepPtr> &time_steps,
                          const std::vector<LgInfo> &lg_infos,
                          const std::vector<shape_secs_t> &shape_secs,
                          bool dynamic_compile) {
  int64_t group_num = lg_infos.size();
  BasicTimeStepPtr up_time_step;
  BasicTimeStepPtr down_time_step;
  // <value, timestep_idx>
  std::vector<std::pair<Value, int64_t>> up_group_overlap_op;
  std::vector<std::pair<Value, int64_t>> down_group_overlap_op;
  // <lmem_addr, lmem_size>
  std::vector<MemBlock> down_to_up_overlap_buffer;
  std::vector<MemBlock> up_to_down_overlap_buffer;
  std::vector<Value> down_to_up_tensor;
  std::vector<Value> up_to_down_tensor;
  MemBlock lmem_locate;
  Value tensor;
  int64_t timestep_idx;

  for (int64_t i = 1; i < group_num; ++i) {
    if (lg_infos[i - 1].group_ops.size() <= 1 ||
        lg_infos[i].group_ops.size() <= 1) {
      continue;
    }
    up_time_step = time_steps[i - 1];
    down_time_step = time_steps[i];
    const LgInfo &up_group = lg_infos[i - 1];
    const LgInfo &down_group = lg_infos[i];
    const shape_secs_t &up_secs = shape_secs[i - 1];
    const shape_secs_t &down_secs = shape_secs[i];
    up_group_overlap_op.clear();
    down_group_overlap_op.clear();
    // find the alternative operation that can be overlapped
    find_alter_overlap_op(up_group, up_time_step, up_secs, up_group_overlap_op,
                          down_group, down_time_step, down_secs,
                          down_group_overlap_op, dynamic_compile);

    // down group to up group overlap
    down_to_up_overlap_buffer.clear();
    down_to_up_tensor.clear();
    // create buffer
    for (size_t j = 0; j < down_group_overlap_op.size(); ++j) {
      tensor = down_group_overlap_op[j].first;
      timestep_idx = down_group_overlap_op[j].second;
      lmem_locate = down_time_step->get_lmem_locate(tensor, timestep_idx);
      down_to_up_overlap_buffer.push_back(lmem_locate);
      down_to_up_tensor.push_back(tensor);
    }

    auto down_to_up_depth =
        up_overlap_depth(up_time_step, down_to_up_overlap_buffer,
                         down_to_up_tensor, up_group, up_secs);
    assign_down_to_up_overlap_timestep(
        up_group, up_secs, up_time_step, down_group, down_secs, down_time_step,
        down_group_overlap_op, down_to_up_depth, down_to_up_overlap_buffer);

    // up group to down group overlap
    up_to_down_overlap_buffer.clear();
    up_to_down_tensor.clear();
    // create buffer
    for (size_t j = 0; j < up_group_overlap_op.size(); ++j) {
      tensor = up_group_overlap_op[j].first;
      timestep_idx = up_group_overlap_op[j].second;
      lmem_locate = up_time_step->get_lmem_locate(tensor, timestep_idx);
      up_to_down_overlap_buffer.push_back(lmem_locate);
      up_to_down_tensor.push_back(tensor);
    }

    auto up_to_down_depth = down_overlap_depth(
        down_time_step, up_to_down_overlap_buffer, up_to_down_tensor);
    assign_up_to_down_overlap_timestep(
        up_group, up_secs, up_time_step, down_group, down_secs, down_time_step,
        up_group_overlap_op, up_to_down_depth, up_to_down_overlap_buffer);
  }
}

//===================================
// Algorithm about group overlap
//===================================
static void layer_group_overlap(std::vector<BasicTimeStepPtr> &time_steps,
                                const std::vector<LgInfo> &lg_infos,
                                const std::vector<shape_secs_t> &shape_secs) {
  direct_group_overlap_schd(time_steps, lg_infos, shape_secs, false);
}

/// The pass of layer group overlap
class GroupDataMoveOverlapPass : public LgPass {
public:
  GroupDataMoveOverlapPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    layer_group_overlap(pass_ir->time_steps, pass_ir->lg_infos,
                        pass_ir->shape_secs);
    return true;
  }
  virtual std::string name() override { return "GroupDataMoveOverlapPass"; }
  virtual std::string brief() override {
    return "Overlap data move between two layer group";
  }
};

std::unique_ptr<LgPass>
CreateGroupDataMoveOverlapPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new GroupDataMoveOverlapPass(options));
}

} // namespace tpu
} // namespace tpu_mlir
