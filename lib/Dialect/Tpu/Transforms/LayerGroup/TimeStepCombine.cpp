//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LmemAllocator.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

static void update_cycle_info(std::vector<int64_t> &total_gdma_cycle_v,
                              std::vector<int64_t> &total_layer_cycle_v,
                              const BasicTimeStepPtr &time_step,
                              const shape_secs_t &shape_secs) {
  bool one_loop = (shape_secs.nsecs * shape_secs.hsecs * shape_secs.dsecs *
                       shape_secs.wsecs ==
                   1);
  int64_t ts_num = time_step->get_timestep_num();
  total_gdma_cycle_v.clear();
  total_layer_cycle_v.clear();
  total_gdma_cycle_v.resize(ts_num, 0);
  total_layer_cycle_v.resize(ts_num, 0);
  ValueIntMap gdma_cycle = time_step->get_gdma_cycle();
  std::map<Operation *, int64_t> layer_cycle = time_step->get_layer_cycle();

  // update cycle for each timestep
  for (int64_t ts = 0; ts < ts_num; ++ts) {
    const TpuTsField &cur_layers = time_step->getLayers(ts);
    for (size_t i = 0; i < cur_layers.size(); ++i) {
      assert(layer_cycle.find(cur_layers[i]) != layer_cycle.end());
      total_layer_cycle_v[ts] += layer_cycle[cur_layers[i]];
    }

    const GdmaTsField &cur_tensors = time_step->getTensors(ts);
    for (size_t i = 0; i < cur_tensors.size(); ++i) {
      auto value = cur_tensors[i].first;
      assert(gdma_cycle.find(value) != gdma_cycle.end());
      // tensor hold in memory just ignore its time cost
      if (time_step->is_tensor_hold_in_lmem(value) && !one_loop) {
        continue;
      }
      total_gdma_cycle_v[ts] += gdma_cycle[value];
    }
  }
}

// move given tensor from src_ts to other timestep,
// and sorted it according to cycle profit
static void
select_valid_dst_timesteps(BasicTimeStepPtr &time_step,
                           const std::vector<int64_t> &total_layer_cycle_v,
                           const std::vector<int64_t> &total_gdma_cycle_v,
                           std::vector<GdmaTsField> ts_tensors_v,
                           std::list<int64_t> &sel_timesteps,
                           GdmaElt &sel_tensor, int64_t src_ts) {
  int64_t ts_num = time_step->get_timestep_num();
  auto &gdma_cycle = time_step->get_gdma_cycle();

  // timesteps_v: (cycle_profit, timestep_idx)
  std::vector<std::map<int64_t, int64_t, std::greater<int64_t>>> timesteps_v;

  int64_t location = -1;
  int64_t best_cycle_profit = 0;
  for (size_t i = 0; i < ts_tensors_v[src_ts].size(); ++i) {
    std::map<int64_t, int64_t, std::greater<int64_t>> sorted_timesteps;
    auto cur_tensor = ts_tensors_v[src_ts][i];
    Value v = cur_tensor.first;
    if (time_step->is_tensor_hold_in_lmem(v)) {
      timesteps_v.push_back(sorted_timesteps);
      continue;
    }
    int64_t src_slack =
        total_layer_cycle_v[src_ts] - total_gdma_cycle_v[src_ts];
    int64_t src_slack_after = std::min(src_slack + gdma_cycle[v], (int64_t)0);
    for (int64_t ts = 0; ts < ts_num; ++ts) {
      if (!time_step->tensor_can_move(cur_tensor, src_ts, ts) || ts == src_ts) {
        continue;
      }
      int64_t dst_slack = total_layer_cycle_v[ts] - total_gdma_cycle_v[ts];
      int64_t dst_slack_after = std::min(dst_slack - gdma_cycle[v], (int64_t)0);
      int64_t cycle_profit = dst_slack_after + src_slack_after - src_slack;

      if (cycle_profit <= 0) {
        continue;
      }
      sorted_timesteps.insert(std::make_pair(cycle_profit, ts));
      if (cycle_profit > best_cycle_profit) {
        best_cycle_profit = cycle_profit;
        sel_tensor = cur_tensor;
        location = i;
      }
    }
    timesteps_v.push_back(sorted_timesteps);
  }
  if (best_cycle_profit != 0 && !timesteps_v[location].empty()) {
    sel_timesteps.push_back(timesteps_v[location].begin()->second);
  }
}

static void update_mem_buffer_by_timestep_merge(
    BasicTimeStepPtr &time_step, std::vector<GdmaTsField> &ts_tensors_v,
    const MemBuff &src_mem_buffer, MemBuff &dst_mem_buffer, int64_t ts,
    bool consider_hold_in_tensor) {
  mem_buffer_key_t mem_key;
  auto is_the_same_value = [&](const GdmaElt &elt) {
    return elt.first == mem_key.value;
  };
  mem_buffer_value_t mem_val;
  // TIMESTEP_LD_ST ldst_type = TIMESTEP_LOAD;
  for (auto it = src_mem_buffer.begin(); it != src_mem_buffer.end(); it++) {
    mem_key = it->first;
    mem_val = it->second;
    if (mem_key.type == LMEM_OPERATION) {
      if (mem_val.start_ts > ts) {
        mem_val.start_ts -= 1;
        mem_val.end_ts -= 1;
      }
    } else {
      mem_val.start_ts = mem_val.start_ts - (mem_val.start_ts > ts ? 1 : 0);
      mem_val.end_ts = mem_val.end_ts - (mem_val.end_ts > ts ? 1 : 0);
      // If software pipline, one tensor may start at ts + 1, and end at ts
      if (it->second.start_ts == it->second.end_ts + 1 &&
          mem_val.start_ts == mem_val.end_ts) {
        mem_val.start_ts =
            (it->second.start_ts >= (time_step->get_timestep_num() - 1)
                 ? 0
                 : it->second.start_ts);
      }
    }

    // if there is hold coeff tensor and make current layer can not merge with
    // back layer, just move hold coeff tensor to the first timestep
    if (mem_key.type != LMEM_OPERATION && consider_hold_in_tensor &&
        time_step->is_tensor_hold_in_lmem(mem_key.value)) {
      if (ts > 0 && it->second.end_ts > it->second.start_ts) {
        auto iter = std::find_if(ts_tensors_v[ts].begin(),
                                 ts_tensors_v[ts].end(), is_the_same_value);
        if (iter != ts_tensors_v[ts].end()) {
          ts_tensors_v[0].push_back(*iter);
          ts_tensors_v[ts].erase(iter);
          mem_val.start_ts = 0;
        }
      }
    }
    dst_mem_buffer[mem_key] = mem_val;
  }
}

static void update_mem_buffer_by_tensor_move(
    BasicTimeStepPtr &time_step, std::vector<GdmaTsField> &ts_tensors_v,
    const MemBuff &src_mem_buffer, MemBuff &dst_mem_buffer,
    const GdmaElt *src_tensors, const int64_t *src_ts, const int64_t *dst_ts,
    int64_t tensor_num) {
  // move tensor from src_ts to dst_ts
  int64_t src_i = 0;
  auto is_the_same_value = [&](const GdmaElt &elt) {
    return elt.first == src_tensors[src_i].first;
  };
  for (int64_t i = 0; i < tensor_num; ++i) {
    src_i = i;
    auto tensor_iter =
        std::find_if(ts_tensors_v[src_ts[i]].begin(),
                     ts_tensors_v[src_ts[i]].end(), is_the_same_value);
    ts_tensors_v[dst_ts[i]].push_back(src_tensors[i]);
    ts_tensors_v[src_ts[i]].erase(tensor_iter);
  }

  // update mem_buffer
  mem_buffer_value_t buffer_value;
  for (auto it = src_mem_buffer.begin(); it != src_mem_buffer.end(); it++) {
    auto &buffer_key = it->first;
    buffer_value = it->second;
    int64_t start_ts = buffer_value.start_ts;
    int64_t end_ts = buffer_value.end_ts;
    for (int64_t i = 0; i < tensor_num; ++i) {
      if (buffer_key.value == src_tensors[i].first &&
          buffer_key.type != LMEM_OPERATION) {
        if (src_tensors[i].second.mode == TIMESTEP_STORE) {
          if ((start_ts < end_ts && dst_ts[i] > end_ts) ||
              (start_ts > end_ts && dst_ts[i] > end_ts &&
               dst_ts[i] < start_ts)) {
            buffer_value.end_ts = dst_ts[i];
          }
        } else if (src_tensors[i].second.mode == TIMESTEP_LOAD) {
          buffer_value.start_ts = dst_ts[i];
        } else {
          llvm_unreachable("Wrong tensor timestep type!");
        }
      }
    }
    dst_mem_buffer[buffer_key] = buffer_value;
  }
}

bool lmem_alloc_by_timestep_merge(BasicTimeStepPtr &time_step, int64_t ts) {
  std::map<int64_t, std::pair<mem_buffer_key_t, int64_t>, std::greater<int64_t>>
      gdma_mem, npu_mem; // <lmem_size, <mem_buffer_key_t, lmem_addr>>
  std::map<int64_t, int64_t> gdma_used_mem,
      npu_used_mem; // lmem_addr, lmem_size
  auto &cur_ts_tensors = time_step->getTensors(ts);
  auto &next_ts_tensors = time_step->getTensors(ts + 1);

  auto is_gdma = [](Value v, GdmaTsField gdma_field) {
    for (auto &iter : gdma_field) {
      if (iter.first == v) {
        return true;
      }
    }
    return false;
  };
  auto &lmem_buffer = time_step->get_lmem_buffer();
  for (auto &iter : lmem_buffer) {
    int64_t start_ts = iter.second.start_ts;
    int64_t end_ts = iter.second.end_ts;
    auto &mem_key = iter.first;
    auto &mem_val = iter.second;
    if (start_ts == ts + 1) {
      if (mem_key.type == LMEM_OPERATION) {
        npu_mem[mem_val.size] = std::make_pair(mem_key, mem_val.addr);
      } else if (!time_step->is_tensor_hold_in_lmem(mem_key.value)) {
        if (is_gdma(mem_key.value, next_ts_tensors)) {
          gdma_mem[mem_val.size] = std::make_pair(mem_key, mem_val.addr);
        } else {
          npu_mem[mem_val.size] = std::make_pair(mem_key, mem_val.addr);
        }
      }
    } else if ((ts >= start_ts && ts <= end_ts) ||
               (start_ts == end_ts && ts == start_ts) ||
               (start_ts > end_ts && (ts >= start_ts || ts < end_ts))) {
      if (mem_key.type == LMEM_OPERATION) {
        npu_used_mem[mem_val.addr] = mem_val.size;
      } else if (is_gdma(mem_key.value, cur_ts_tensors)) {
        gdma_used_mem[mem_val.addr] = mem_val.size;
      } else {
        npu_used_mem[mem_val.addr] = mem_val.size;
      }
    }
    if (mem_key.type != LMEM_OPERATION &&
        time_step->is_tensor_hold_in_lmem(mem_key.value)) {
      gdma_used_mem[mem_val.addr] = mem_val.size;
    }
  }

  std::set<int64_t> used_banks;
  auto update_mem_bank = [&used_banks](std::map<int64_t, int64_t> &used_mem) {
    for (auto &iter : used_mem) {
      int64_t bank_id = iter.first / Arch::LMEM_BANK_BYTES;
      used_banks.insert(bank_id);
      if (used_mem.find(bank_id * Arch::LMEM_BANK_BYTES) == used_mem.end()) {
        used_mem[bank_id * Arch::LMEM_BANK_BYTES] = 0;
      }
    }
  };

  update_mem_bank(gdma_used_mem);
  update_mem_bank(npu_used_mem);
  std::list<int64_t> unused_banks;
  for (int64_t i = 0; i < Arch::LMEM_BANKS; ++i) {
    if (used_banks.count(i) == 0) {
      unused_banks.push_back(i);
    }
  }

  std::map<mem_buffer_key_t, int64_t> reassign_buffers;
  auto assign_addr = [&](std::map<int64_t, std::pair<mem_buffer_key_t, int64_t>,
                                  std::greater<int64_t>> &mem_map,
                         std::map<int64_t, int64_t> &used_mem_map,
                         size_t &cnt) {
    for (auto &iter : mem_map) {
      auto in_iter = used_mem_map.begin();
      for (; in_iter != used_mem_map.end();) {
        auto pre_iter = in_iter++;
        if (in_iter == used_mem_map.end()) {
          break;
        }
        int64_t bound = 0;
        if (in_iter == used_mem_map.end()) {
          bound = (pre_iter->first / Arch::LMEM_BANK_BYTES + 1) *
                  Arch::LMEM_BANK_BYTES;
        } else {
          bound = in_iter->first / Arch::LMEM_BANK_BYTES ==
                          pre_iter->first / Arch::LMEM_BANK_BYTES
                      ? in_iter->first
                      : (pre_iter->first / Arch::LMEM_BANK_BYTES + 1) *
                            Arch::LMEM_BANK_BYTES;
        }
        int64_t offset =
            align_up(pre_iter->first + pre_iter->second, Arch::EU_BYTES);
        if (iter.first + offset <= bound) {
          reassign_buffers[iter.second.first] = offset;
          used_mem_map[offset] = iter.first;
          cnt++;
          break;
        }
      }
      if (!unused_banks.empty() &&
          reassign_buffers.count(iter.second.first) == 0) {
        int64_t offset = unused_banks.front() * Arch::LMEM_BANK_BYTES;
        if (offset + iter.first <=
            (unused_banks.front() + 1) * Arch::LMEM_BANK_BYTES) {
          reassign_buffers[iter.second.first] = offset;
          used_mem_map[offset] = iter.first;
          unused_banks.pop_front();
          cnt++;
          break;
        }
      }
    }
  };

  size_t cnt0 = 0, cnt1 = 0;
  assign_addr(npu_mem, npu_used_mem, cnt0);
  assign_addr(gdma_mem, gdma_used_mem, cnt0);
  bool res = (cnt0 == npu_mem.size() && cnt1 == gdma_mem.size());
  if (res) {
    for (auto &iter : reassign_buffers) {
      time_step->set_lmem_addr(iter.first, iter.second);
    }
  }
  return res;
}

/* merge two timesteps;
 * If total gdma cycle > total bdc cycle in current timestep or next timestep,
 * and then merge these two timestep
 */
static void merge_timesteps(const LgInfo &lg_info, BasicTimeStepPtr &time_step,
                            const shape_secs_t &shape_secs,
                            std::vector<TpuTsField> &ts_layers_v,
                            std::vector<GdmaTsField> &ts_tensors_v,
                            std::vector<int64_t> &total_layer_cycle_v,
                            std::vector<int64_t> &total_gdma_cycle_v,
                            MemBuff &mem_buffer, bool &print_log,
                            int64_t group_idx, const LgOptions &options) {
  auto lmem_allocator = LmemAllocator(options);
  MemBuff p_lmem = time_step->get_lmem_buffer();
  std::vector<TpuTsField> p_layers(ts_layers_v);
  std::vector<GdmaTsField> p_tensors(ts_tensors_v);
  std::vector<std::string> ss;
  // cycle may have error with real time cost
  float error = 0.1f;

  int64_t ts = -1;
  std::set<int64_t> exclude_ts;
  while (true) {
    ts++;
    int64_t ts_num = time_step->get_timestep_num();
    if (ts == ts_num - 1) {
      break;
    }
    if (exclude_ts.count(ts) != 0) {
      continue;
    }
    float cur_layer_cycle = total_layer_cycle_v[ts];
    float cur_gdma_cycle = total_gdma_cycle_v[ts];
    float next_layer_cycle = total_layer_cycle_v[ts + 1];
    float next_gdma_cycle = total_gdma_cycle_v[ts + 1];
    if (cur_gdma_cycle <= cur_layer_cycle &&
        next_gdma_cycle <= next_layer_cycle &&
        (cur_layer_cycle - cur_gdma_cycle) / cur_layer_cycle > error &&
        (next_layer_cycle - next_gdma_cycle) / next_layer_cycle > error) {
      continue;
    }

    // info of timestep
    std::vector<GdmaTsField> new_ts_tensors_v(ts_tensors_v);
    std::vector<TpuTsField> new_ts_layers_v(ts_layers_v);
    MemBuff new_mem_buffer;

    bool one_loop = (shape_secs.nsecs * shape_secs.hsecs * shape_secs.dsecs *
                         shape_secs.wsecs ==
                     1);
    if (time_step->layer_can_merge_backward(ts, one_loop)) {
      // update timestep layers and tensors
      new_ts_layers_v[ts].insert(new_ts_layers_v[ts].end(),
                                 new_ts_layers_v[ts + 1].begin(),
                                 new_ts_layers_v[ts + 1].end());
      new_ts_layers_v.erase(new_ts_layers_v.begin() + ts + 1);
      new_ts_tensors_v[ts].insert(new_ts_tensors_v[ts].end(),
                                  new_ts_tensors_v[ts + 1].begin(),
                                  new_ts_tensors_v[ts + 1].end());
      new_ts_tensors_v.erase(new_ts_tensors_v.begin() + ts + 1);

      if (!lmem_alloc_by_timestep_merge(time_step, ts)) {
        continue;
      }

      // update membuffer timestep
      update_mem_buffer_by_timestep_merge(time_step, new_ts_tensors_v,
                                          time_step->get_lmem_buffer(),
                                          new_mem_buffer, ts, !one_loop);

      time_step->reset_timestep(new_ts_layers_v, new_ts_tensors_v,
                                new_mem_buffer);
      ts_layers_v = new_ts_layers_v;
      ts_tensors_v = new_ts_tensors_v;
      new_mem_buffer = mem_buffer;
      update_cycle_info(total_gdma_cycle_v, total_layer_cycle_v, time_step,
                        shape_secs);
      if (print_log) {
        ss.push_back("===group idx: " + std::to_string(group_idx));
      }
      ss.push_back("merge timestep " + std::to_string(ts + 1) +
                   " to timestep " + std::to_string(ts));
      print_log = false;
      auto iter = exclude_ts.find(ts - 1);
      if (iter != exclude_ts.end()) {
        exclude_ts.erase(iter);
      }
      ts = -1;
    }
  }

  if (lmem_allocator.assignLmemAddr(lg_info, time_step, shape_secs) == false) {
    ss.clear();
    time_step->reset_timestep(p_layers, p_tensors, p_lmem);
  }
  for (auto iter : ss) {
    LAYER_GROUP_LOG_DEBUG_BLOCK({ llvm::outs() << iter << "\n"; });
  }
}

/* reassign timestep tensors;
 * 1. If one timestep's total gdma cycle > total bdc cycle,
 * then move tensor to other timestep to gain best profit.
 * 2. Swap tensors for store and load if their profit is equal, to make load as
 * last as possible, and make store as soon as possible.
 */
static void reassign_timestep_tensors(
    const LgInfo &lg_info, BasicTimeStepPtr &time_step,
    const shape_secs_t &shape_secs, std::vector<TpuTsField> &ts_layers_v,
    std::vector<GdmaTsField> &ts_tensors_v,
    std::vector<int64_t> &total_layer_cycle_v,
    std::vector<int64_t> &total_gdma_cycle_v, MemBuff &mem_buffer,
    bool &print_log, int64_t group_idx, const LgOptions &options) {
  // move tensor to other timestep if gdma_cycle > bdc_cycle
  auto lmem_allocator = LmemAllocator(options);
  int64_t ts = -1;
  GdmaElt sel_tensor;
  std::list<int64_t> sel_timesteps;
  std::set<int64_t> exclude_timesteps;
  int64_t ts_num = time_step->get_timestep_num();
  while (ts < ts_num - 1) {
    ++ts;
    if (total_gdma_cycle_v[ts] <= total_layer_cycle_v[ts] ||
        exclude_timesteps.count(ts) != 0) {
      continue;
    }
    select_valid_dst_timesteps(time_step, total_layer_cycle_v,
                               total_gdma_cycle_v, ts_tensors_v, sel_timesteps,
                               sel_tensor, ts);

    int64_t src_ts = ts;
    while (!sel_timesteps.empty()) {
      int64_t dst_ts = sel_timesteps.front();
      sel_timesteps.pop_front();

      // update timestep tensors info
      std::vector<GdmaTsField> tmp_tensors(ts_tensors_v);
      MemBuff new_mem_buffer;
      update_mem_buffer_by_tensor_move(time_step, tmp_tensors, mem_buffer,
                                       new_mem_buffer, &sel_tensor, &src_ts,
                                       &dst_ts, 1);
      time_step->reset_timestep(ts_layers_v, tmp_tensors, new_mem_buffer);
      if (lmem_allocator.assignLmemAddr(lg_info, time_step, shape_secs)) {
        // update timestep layers and tensors
        ts_tensors_v = tmp_tensors;
        exclude_timesteps.insert(dst_ts);

        // update timestep mem buffer
        mem_buffer = time_step->get_lmem_buffer();
        update_cycle_info(total_gdma_cycle_v, total_layer_cycle_v, time_step,
                          shape_secs);
        if (print_log) {
          LAYER_GROUP_LOG_DEBUG_BLOCK(
              { llvm::outs() << "===group_idx: " << group_idx; });
        }
        LAYER_GROUP_LOG_DEBUG_BLOCK({
          llvm::outs() << "move tensor " << module::getName(sel_tensor.first)
                       << " from timestep " << src_ts << " to timestep "
                       << dst_ts;
        });
        print_log = false;
        break;
      } else {
        // if can not combine two timesteps, reset all
        time_step->reset_timestep(ts_layers_v, ts_tensors_v, mem_buffer);
      }
    }
    if (sel_timesteps.empty() || ts != -1) {
      exclude_timesteps.insert(src_ts);
    }
  }

  // swap load and store gdma operation, let store go forward, let load go
  // backward
  auto gdma_cycle = time_step->get_gdma_cycle();
  for (int64_t src_ts = 0; src_ts < ts_num; src_ts++) {
    GdmaElt src_tensor, dst_tensor;
    bool found_src_tensor = false;
    bool found_dst_tensor = false;
    // get tensor_load that will be moved from src_ts to dst_ts
    for (size_t j = 0; j < ts_tensors_v[src_ts].size(); ++j) {
      if (!time_step->is_tensor_hold_in_lmem(ts_tensors_v[src_ts][j].first) &&
          ts_tensors_v[src_ts][j].second.mode == TIMESTEP_LOAD) {
        src_tensor = ts_tensors_v[src_ts][j];
        found_src_tensor = true;
        break;
      }
    }
    if (!found_src_tensor) {
      continue;
    }
    // get tensor_store that will be moved from dst_ts to src_ts
    int64_t dst_ts = -1;
    for (dst_ts = ts_num - 1; dst_ts > src_ts; --dst_ts) {
      if (!time_step->tensor_can_move(src_tensor, src_ts, dst_ts)) {
        continue;
      }
      for (size_t k = 0; k < ts_tensors_v[dst_ts].size(); ++k) {
        if (ts_tensors_v[dst_ts][k].second.mode != TIMESTEP_STORE ||
            !time_step->tensor_can_move(ts_tensors_v[dst_ts][k], dst_ts,
                                        src_ts)) {
          continue;
        }
        int64_t src_tensor_cycle = gdma_cycle[src_tensor.first];
        int64_t dst_tensor_cycle = gdma_cycle[ts_tensors_v[dst_ts][k].first];
        int64_t src_slack =
            total_layer_cycle_v[src_ts] - total_gdma_cycle_v[src_ts];
        int64_t dst_slack =
            total_layer_cycle_v[dst_ts] - total_gdma_cycle_v[dst_ts];
        int64_t src_slack_after = std::min(
            src_slack - dst_tensor_cycle + src_tensor_cycle, (int64_t)0);
        int64_t dst_slack_after = std::min(
            dst_slack + dst_tensor_cycle - src_tensor_cycle, (int64_t)0);
        src_slack = src_slack >= 0 ? 0 : src_slack;
        dst_slack = dst_slack >= 0 ? 0 : dst_slack;
        if (src_slack_after + dst_slack_after < src_slack + dst_slack) {
          continue;
        }
        dst_tensor = ts_tensors_v[dst_ts][k];
        found_dst_tensor = true;
        break;
      }
      if (found_dst_tensor) {
        break;
      }
    }
    if (!found_dst_tensor) {
      continue;
    }

    // update timestep tensors info
    std::vector<GdmaTsField> new_ts_tensors_v(ts_tensors_v);
    MemBuff new_mem_buffer;
    GdmaElt src_tensor_v[2] = {src_tensor, dst_tensor};
    int64_t src_ts_v[2] = {src_ts, dst_ts};
    int64_t dst_ts_v[2] = {dst_ts, src_ts};
    update_mem_buffer_by_tensor_move(time_step, new_ts_tensors_v, mem_buffer,
                                     new_mem_buffer, src_tensor_v, src_ts_v,
                                     dst_ts_v, 2);

    time_step->reset_timestep(ts_layers_v, new_ts_tensors_v, new_mem_buffer);
    if (lmem_allocator.assignLmemAddr(lg_info, time_step, shape_secs)) {
      // update timestep info
      ts_tensors_v = std::move(new_ts_tensors_v);
      mem_buffer = time_step->get_lmem_buffer();
      update_cycle_info(total_gdma_cycle_v, total_layer_cycle_v, time_step,
                        shape_secs);
      if (print_log) {
        LAYER_GROUP_LOG_DEBUG_BLOCK(
            { llvm::outs() << "===group idx: " << group_idx; });
      }
      LAYER_GROUP_LOG_DEBUG_BLOCK({
        llvm::outs() << "move tensor " << module::getName(src_tensor.first)
                     << " from timestep " << src_ts << " to timestep "
                     << dst_ts;
        llvm::outs() << "move tensor " << module::getName(dst_tensor.first)
                     << " from timestep " << dst_ts << " to timestep "
                     << src_ts;
      });
      print_log = false;
      break;
    } else {
      // if lmem allocate failed, reset all
      time_step->reset_timestep(ts_layers_v, ts_tensors_v, mem_buffer);
    }
  }
}

// search each timestep, if gdma time is less than bdc time, do nothing
// else reassign tensors first, and then merge timestep
static void memory_aware_timestep_combine(const LgInfo &lg_info,
                                          BasicTimeStepPtr &time_step,
                                          const shape_secs_t &shape_secs,
                                          int64_t group_idx,
                                          const LgOptions &options) {
  if (lg_info.group_ops.size() == 1) {
    return;
  }

  std::vector<TpuTsField> ts_layers_v;
  std::vector<GdmaTsField> ts_tensors_v;
  const auto &timestep_table = time_step->get_timestep_table();
  for (auto &row : timestep_table) {
    ts_layers_v.push_back(row.tpu0_ts_field);
    ts_tensors_v.push_back(row.gdma0_ts_field);
  }

  std::vector<int64_t> total_gdma_cycle_v;
  std::vector<int64_t> total_layer_cycle_v;
  update_cycle_info(total_gdma_cycle_v, total_layer_cycle_v, time_step,
                    shape_secs);

  bool print_log = true;
  MemBuff mem_buffer(time_step->get_lmem_buffer());
  mem_buffer.insert(time_step->get_l2mem_buffer().begin(),
                    time_step->get_l2mem_buffer().end());
  // lower performance gain, but may result in larger compile time cost
  if (lg_info.group_ops.size() <= 100) {
    reassign_timestep_tensors(lg_info, time_step, shape_secs, ts_layers_v,
                              ts_tensors_v, total_layer_cycle_v,
                              total_gdma_cycle_v, mem_buffer, print_log,
                              group_idx, options);
  }
  merge_timesteps(lg_info, time_step, shape_secs, ts_layers_v, ts_tensors_v,
                  total_layer_cycle_v, total_gdma_cycle_v, mem_buffer,
                  print_log, group_idx, options);
}

static void timestep_combine(const std::vector<LgInfo> &lg_infos,
                             std::vector<BasicTimeStepPtr> &time_steps,
                             const std::vector<shape_secs_t> &shape_secs,
                             const LgOptions &options) {
  assert(lg_infos.size() == time_steps.size());
  assert(lg_infos.size() == shape_secs.size());
  for (size_t i = 0; i < lg_infos.size(); ++i) {
    memory_aware_timestep_combine(lg_infos[i], time_steps[i], shape_secs[i], i,
                                  options);
  }
}

/// The pass for time step combine
class TimeStepCombinePass : public LgPass {
public:
  TimeStepCombinePass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    timestep_combine(pass_ir->lg_infos, pass_ir->time_steps,
                     pass_ir->shape_secs, options_);
    return true;
  }
  virtual std::string name() override { return "TimeStepCombinePass"; }
  virtual std::string brief() override {
    return "Combine time step for better parallel balance";
  }
};

std::unique_ptr<LgPass> CreateTimeStepCombinePass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new TimeStepCombinePass(options));
}

} // namespace tpu
} // namespace tpu_mlir
