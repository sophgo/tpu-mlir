//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/TimeStepMethod.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

namespace tpu_mlir {
namespace tpu {

void TimeStepMethod::layer_nearest_timestep_assignment(BasicTimeStep *time_step,
                                                       TensorInfo &tensor_infos,
                                                       const LgInfo &lg_info) {
  const std::vector<Operation *> &group_ops = lg_info.group_ops;
  bool have_load_tensor;
  TpuTsField tpu_field;
  GdmaTsField gdma_field;
  std::set<Value, value_compare> tensor_in_lmem;

  Operation *op;
  tensor_info_t tensor_info;
  // in nearest algorithm, each op calculation will be assigned to a timestep
  for (size_t i = 0; i < group_ops.size(); ++i) {
    op = group_ops[i];
    // timestep: 0
    // load current layer's input from lmem
    // current layer will have tpu_field in next timestep
    DEBUG_WITH_TYPE("timestep_assign", {
      llvm::dbgs() << "; action = layer_nearest_timestep_assignment"
                   << "; ts = " << i << "\n";
    });
    if (i == 0) {
      // stage 0, only have load timestep
      gdma_field.clear();
      have_load_tensor = false;
      for (auto in : op->getOperands()) {
        if (in.getType().isa<NoneType>()) {
          continue;
        }
        if (tensor_in_lmem.count(in) == 0) {
          auto iter = tensor_infos.find(in);
          if (iter != tensor_infos.end()) {
            tensor_info = iter->second;
          }
          tensor_info.mode = TIMESTEP_LOAD;
          gdma_field.push_back(std::make_pair(in, tensor_info));
          have_load_tensor = true;
          tensor_in_lmem.insert(in);
        }
      }
      if (have_load_tensor) {
        time_step->add_gdma0_ts_field(gdma_field);
      }
    }

    tpu_field.clear();
    gdma_field.clear();

    // stage 1, in pipeline, all calculate, load, store ops are in the same
    // timestep
    for (auto out : get_output_values(op)) {
      tensor_in_lmem.insert(out);
    }
    // stage 1.1: add current gdma and tpu timestep
    tpu_field.push_back(op);

    // layer: [1, N-1)
    // stage 1.1: pre load next layer's input in current timestep
    if (i != group_ops.size() - 1) {
      auto next_op = group_ops[i + 1];
      for (auto next_in : next_op->getOperands()) {
        if (next_in.getType().isa<NoneType>()) {
          continue;
        }
        if (tensor_in_lmem.count(next_in) == 0) {
          auto iter = tensor_infos.find(next_in);
          if (iter != tensor_infos.end()) {
            tensor_info = iter->second;
          }
          tensor_info.mode = TIMESTEP_LOAD;
          gdma_field.push_back(std::make_pair(next_in, tensor_info));
          tensor_in_lmem.insert(next_in);
        }
      }
    }

    // layer: [1, N-1)
    // stage 1.2: store current layer's output to lmem
    if (i > 0) {
      auto pre_op = group_ops[i - 1];
      for (auto pre_out : get_output_values(pre_op)) {
        if (std::find(lg_info.group_outs.begin(), lg_info.group_outs.end(),
                      pre_out) != lg_info.group_outs.end()) {
          tensor_info = tensor_infos[pre_out];
          tensor_info.mode = TIMESTEP_STORE;
          gdma_field.push_back(std::make_pair(pre_out, tensor_info));
        }
      }
    }

    // add current gdma and tpu timestep
    // stage 1 finally add
    if (!(tpu_field.empty() && gdma_field.empty())) {
      time_step->add_tpu0_gdma0_ts_field(tpu_field, gdma_field);
    }

    // last layer
    // store last layer's output to lmem in a new timestep
    // stage 2: last layer will only have gdma_field
    if (i == group_ops.size() - 1) {
      gdma_field.clear();
      for (auto out : get_output_values(op)) {
        tensor_info = tensor_infos[out];
        tensor_info.mode = TIMESTEP_STORE;
        gdma_field.push_back(std::make_pair(out, tensor_info));
      }
      time_step->add_gdma0_ts_field(gdma_field);
    }
  }

  GROUP_DEBUG_WITH_TYPE("timestep_assign", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "timestep_assign", "intermediate_result",
                        "using nearest algorithm to assign timestep for tpu "
                        "and gdma operations firstly")
                 << "\n============= nearest algorithm =============\n";
    time_step->show_timestep_table();
  });

  // use software pipeline
  if (group_ops.size() > 1) {
    time_step->software_pipeline();
  }

  GROUP_DEBUG_WITH_TYPE("timestep_assign", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "timestep_assign", "intermediate_result",
                        "adust timesteps considering the software pipeline ")
                 << "\n============= software pipeline =============\n";
    time_step->show_timestep_table();
  });
}

bool is_tensor_accessed_by_npu(Value v, BasicTimeStep *time_step, int64_t ts) {
  auto &ts_layers = time_step->getLayers(ts);
  for (auto op : ts_layers) {
    for (auto in : op->getOperands()) {
      if (v == in) {
        return true;
      }
    }
    for (auto out : get_output_values(op)) {
      if (v == out) {
        return true;
      }
    }
  }
  return false;
}

bool TimeStepMethod::process(BasicTimeStep *time_step, TensorInfo &tensor_infos,
                             const LgInfo &lg_info,
                             const shape_secs_t &shape_secs, bool gen_idx) {
  // backward update slice
  if (gen_idx) {
    GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
      llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                          "stripe_mine_idx_slice", "call_function",
                          "backward and update slice_info of tensors starting "
                          "from output tensors "
                          "according to shape_secs, store the idx and slice of "
                          "each tile in `tensor_infos`")
                   << "\n";
    });
    if (stripe_mine_idx_slice(lg_info, shape_secs, tensor_infos, options_) ==
        false) {
      return false;
    }
  } else {
    if (stripe_mine_max_slice(lg_info, shape_secs, tensor_infos, options_) ==
        false) {
      return false;
    }
  }

  // update tensor_infos
  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "update_tensor_infos", "call_function",
                        "set tags and slice_info for specific tensors")
                 << "\n";
  });
  update_tensor_infos(lg_info, tensor_infos, shape_secs);

  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "memory_aware_timestep_assignment", "call_function",
                        "assign timesteps and try to optimize the performance "
                        "by moving timesteps")
                 << "\n";
  });
  // layer_nearest_timestep_assignment(time_step, tensor_infos, lg_info);
  memory_aware_timestep_assignment(time_step, tensor_infos, lg_info);

  GROUP_DEBUG_WITH_TYPE("lg_step", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "gen_hold_coeff", "call_function",
                        "use map hold_coeff_ to store whether we needs the "
                        "tensor to hold in lmem"
                        "for better performance")
                 << "\n";
  });
  time_step->gen_hold_coeff();
  return true;
}

void TimeStepMethod::bubble_tensor_to_best_ts(
    std::list<GdmaElt>::iterator sel_list_iter, int64_t cur_ts, int64_t best_ts,
    BasicTimeStep *time_step, ValueIntMap &tensor_to_cycle,
    ValueIntMap &tensor_to_bufsize,
    std::vector<std::list<GdmaElt>> &tensor_timesteps,
    std::vector<int64_t> &timestep_cycle_slack) {
  // bubble the selected tensor to the right ts
  bool find_best = false;
  Value tensor;
  auto best_sel_tensor = sel_list_iter->first;
  auto gdma_type = sel_list_iter->second.mode;
  int64_t cur_profit = 0, max_profit = 0;
  int64_t cur_new_slack, dst_cost;

  bool is_valid;
  int64_t pre_ts = cur_ts;
  int64_t next_ts = get_next_ts(is_valid, cur_ts, gdma_type, best_ts);
  while (is_valid) {
    tensor_timesteps[next_ts].push_back(*sel_list_iter);
    timestep_cycle_slack[next_ts] -= tensor_to_cycle[best_sel_tensor];
    tensor_timesteps[pre_ts].erase(sel_list_iter);
    timestep_cycle_slack[pre_ts] += tensor_to_cycle[best_sel_tensor];

    if (next_ts == best_ts) {
      break;
    }

    if (gdma_type == TIMESTEP_STORE &&
        is_tensor_accessed_by_npu(best_sel_tensor, time_step, next_ts)) {
      sel_list_iter = tensor_timesteps[next_ts].end();
      sel_list_iter--;
    } else {
      max_profit = 0;
      find_best = false;
      for (auto list_iter = tensor_timesteps[next_ts].begin();
           list_iter != tensor_timesteps[next_ts].end(); ++list_iter) {
        if (gdma_type != list_iter->second.mode) {
          continue;
        }
        tensor = list_iter->first;
        int64_t new_range_end =
            time_step->get_tensor_range_end(*list_iter, next_ts);
        if ((is_timestep_load(gdma_type) && new_range_end > best_ts) ||
            (gdma_type == TIMESTEP_STORE && new_range_end < best_ts) ||
            (gdma_type == TIMESTEP_STORE &&
             is_tensor_accessed_by_npu(tensor, time_step, best_ts))) {
          continue;
        }
        cur_new_slack = timestep_cycle_slack[next_ts] + tensor_to_cycle[tensor];
        cur_profit = std::min(cur_new_slack, (int64_t)0) -
                     std::min(timestep_cycle_slack[next_ts], (int64_t)0);

        dst_cost = timestep_cycle_slack[best_ts] - tensor_to_cycle[tensor];
        dst_cost = dst_cost >= 0 ? 0 : dst_cost;
        cur_profit = cur_profit + dst_cost;
        if (cur_profit > max_profit ||
            (cur_profit == max_profit &&
             tensor_to_bufsize[tensor] < tensor_to_bufsize[best_sel_tensor])) {
          sel_list_iter = list_iter;
          max_profit = cur_profit;
          best_sel_tensor = tensor;
          find_best = true;
        }
      }
      if (find_best == false) {
        break;
      }
    }
    pre_ts = next_ts;
    next_ts = get_next_ts(is_valid, next_ts, gdma_type, best_ts);
  }
}

void TimeStepMethod::memory_aware_timestep_assignment(BasicTimeStep *time_step,
                                                      TensorInfo &tensor_infos,
                                                      const LgInfo &lg_info) {
  layer_nearest_timestep_assignment(time_step, tensor_infos, lg_info);
  if (lg_info.group_ops.size() <= 1) {
    return;
  }
  int64_t timestep_num = time_step->get_timestep_num();
  if (timestep_num == 0) {
    return;
  }
  std::vector<int64_t> timestep_cycle_slack(timestep_num, 0);
  ValueIntMap tensor_to_cycle;
  ValueIntMap tensor_to_bufsize;
  std::vector<std::list<GdmaElt>> tensor_timesteps;

  GROUP_DEBUG_WITH_TYPE("timestep_assign", lg_info, [&]() {
    llvm::dbgs() << "============= memory aware algorithm =============\n";
  });

// remove it after pid_node is extracted
#pragma omp critical(get_cycle)
  get_timestep_cycle_slack(time_step, lg_info, tensor_to_cycle,
                           tensor_to_bufsize, tensor_timesteps,
                           timestep_cycle_slack);

  std::list<GdmaElt>::iterator sel_list_iter;
  int64_t best_ts = 0;
  for (int64_t cur_ts = 0; cur_ts < timestep_num;) {
    int64_t cur_slack = timestep_cycle_slack[cur_ts];
    if (cur_slack >= 0) {
      ++cur_ts;
      continue;
    }

    best_ts = get_best_ts(time_step, lg_info, cur_ts, tensor_to_cycle,
                          tensor_to_bufsize, tensor_timesteps,
                          timestep_cycle_slack, sel_list_iter);
    if (best_ts == -1) {
      ++cur_ts;
      continue;
    }

    bubble_tensor_to_best_ts(sel_list_iter, cur_ts, best_ts, time_step,
                             tensor_to_cycle, tensor_to_bufsize,
                             tensor_timesteps, timestep_cycle_slack);
  }

  // update time_step gdma field
  for (size_t ts = 0; ts < tensor_timesteps.size(); ++ts) {
    GdmaTsField new_tensor_timestep;
    for (auto &iter : tensor_timesteps[ts]) {
      new_tensor_timestep.push_back(std::move(iter));
    }
    time_step->update_gdma0_ts_field(ts, new_tensor_timestep);
  }

  GROUP_DEBUG_WITH_TYPE("timestep_assign", lg_info, [&]() {
    llvm::dbgs() << DEBUGGER_DEFAULT_INFO(
                        "timestep_assign", "final_result",
                        "optimize timesteps using algorithm based on cycle "
                        "slack and buffer area")
                 << "\n============= timestep optimized =============\n";
    time_step->show_timestep_table();
  });

  GROUP_DEBUG_WITH_TYPE("timestep_assign", lg_info, [&]() {
    llvm::dbgs() << "=======================================\n";
  });
}

void TimeStepMethod::get_timestep_cycle_slack(
    BasicTimeStep *time_step, const LgInfo &lg_info,
    ValueIntMap &tensor_to_cycle, ValueIntMap &tensor_to_bufsize,
    std::vector<std::list<GdmaElt>> &tensor_timesteps,
    std::vector<int64_t> &timestep_cycle_slack) {
  int64_t timestep_num = time_step->get_timestep_num();
  auto &tensor_infos = time_step->get_tensor_infos();
  time_step->clear_gdma_cycle();
  time_step->clear_layer_cycle();
  int64_t tensor_cycle = 0;
  int64_t buffer_size = 0;
  for (int64_t ts = 0; ts < timestep_num; ++ts) {
    const auto &ts_layers = time_step->getLayers(ts);
    for (auto op : ts_layers) {
      int64_t cycle_slack = cycle_calculator_->getLocalLayerCycle(
          op, tensor_infos, lg_info.type, true);
      timestep_cycle_slack[ts] += cycle_slack;
      time_step->set_layer_cycle(op, cycle_slack);
    }

    std::list<GdmaElt> list_tensors;
    auto &ts_tensors = time_step->getTensors(ts);
    for (auto &tensor : ts_tensors) {
      auto v = tensor.first;
      auto &ti = tensor.second;
      tensor_cycle = cycle_calculator_->getGdmaCycle(v, ti, lg_info.type);
      buffer_size = get_buffer_size(v, ti, lg_info.type);
      tensor_to_cycle[v] = tensor_cycle;
      tensor_to_bufsize[v] = buffer_size;
      list_tensors.push_back(tensor);
      timestep_cycle_slack[ts] -= tensor_cycle;
      time_step->set_gdma_cycle(tensor.first, tensor_cycle);
    }
    tensor_timesteps.push_back(std::move(list_tensors));
  }
}

int64_t TimeStepMethod::get_next_ts(bool &is_valid, int64_t cur_ts,
                                    TIMESTEP_LD_ST ld_st, int64_t range_end) {
  int64_t next_ts = 0;
  if (is_timestep_load(ld_st)) {
    next_ts = cur_ts - 1;
    is_valid = next_ts >= range_end;
  } else {
    next_ts = cur_ts + 1;
    is_valid = next_ts <= range_end;
  }

  return next_ts;
}

int64_t
TimeStepMethod::get_best_ts(BasicTimeStep *time_step, const LgInfo &lg_info,
                            int64_t cur_ts, ValueIntMap &tensor_to_cycle,
                            ValueIntMap &tensor_to_bufsize,
                            std::vector<std::list<GdmaElt>> &tensor_timesteps,
                            std::vector<int64_t> &timestep_cycle_slack,
                            std::list<GdmaElt>::iterator &sel_list_iter) {
  int64_t src_profit = 0, dst_cost = 0;
  int64_t cur_slack = timestep_cycle_slack[cur_ts];
  int64_t cur_profit = 0, max_profit = 0;
  int64_t cur_area = 0, best_area = 0;
  int64_t best_ts = -1;
  Value best_sel_tensor;
  bool is_valid;
  for (auto list_iter = tensor_timesteps[cur_ts].begin();
       list_iter != tensor_timesteps[cur_ts].end(); ++list_iter) {
    auto tensor = list_iter->first;
    int64_t cur_new_slack = cur_slack + tensor_to_cycle[tensor];
    src_profit = (cur_new_slack >= 0 ? 0 : cur_new_slack) - cur_slack;

    auto gdma_type = list_iter->second.mode;
    int64_t range_end = time_step->get_tensor_range_end(*list_iter, cur_ts);
    int64_t next_ts = get_next_ts(is_valid, cur_ts, gdma_type, range_end);

    while (is_valid) {
      if (gdma_type == TIMESTEP_STORE &&
          is_tensor_accessed_by_npu(tensor, time_step, next_ts)) {
        next_ts = get_next_ts(is_valid, next_ts, gdma_type, range_end);
        continue;
      }
      if (timestep_cycle_slack[next_ts] > 0) {
        dst_cost = timestep_cycle_slack[next_ts] - tensor_to_cycle[tensor];
        dst_cost = dst_cost >= 0 ? 0 : dst_cost;
        cur_area = tensor_to_bufsize[tensor] * (std::abs(next_ts - cur_ts) + 1);
        cur_profit = src_profit + dst_cost;
        if (cur_profit > max_profit) {
          max_profit = cur_profit;
          best_ts = next_ts;
          sel_list_iter = list_iter;
          best_sel_tensor = tensor;
          best_area = cur_area;
        } else if (cur_profit == max_profit && best_ts != -1) {
          if (cur_area < best_area) {
            max_profit = cur_profit;
            best_ts = next_ts;
            sel_list_iter = list_iter;
            best_sel_tensor = tensor;
            best_area = cur_area;
          }
        }
      }
      next_ts = get_next_ts(is_valid, next_ts, gdma_type, range_end);
    }
  }
  return best_ts;
}

// void avoid_bank_conflict_in_timestep(BasicTimeStep *time_step) {
//   int64_t timestep_num = time_step->get_timestep_num();
//   std::vector<Value> conflict_tensors;
//   for (int64_t ts = 0; ts < timestep_num; ++ts) {
//     auto &ts_tensors = time_step->getTensors(ts);
//     for (auto tensor : ts_tensors) {
//       if (tensor.second.mode == TIMESTEP_STORE &&
//       is_tensor_accessed_by_npu(tensor.first, time_step, ts)) {
//         conflict_tensors.push_back(tensor.fisrt);
//       }
//     }
//
//   }
// }

class TimeStepAssignmentPass : public LgPass {
public:
  TimeStepAssignmentPass(const LgOptions &options) { options_ = options; }
  virtual bool run(LgPassIR *pass_ir) override {
    pass_ir->time_steps.clear();
    for (size_t i = 0; i < pass_ir->lg_infos.size(); ++i) {
      auto time_step = std::make_shared<BasicTimeStep>(options_);
      shape_secs_t shape_secs;
      std::vector<std::pair<Value, int64_t>> value_size;
      if (!init_group_data_secs(pass_ir->lg_infos[i], shape_secs, value_size,
                                options_)) {
        return false;
      }
      bool ret =
          time_step->assignTimeStep(pass_ir->lg_infos[i], shape_secs, true);
      if (!ret) {
        llvm::errs() << "time step assign failed for group " << i << "\n";
        return false;
      }
      pass_ir->time_steps.emplace_back(time_step);
      pass_ir->shape_secs.emplace_back(shape_secs);
    }
    return true;
  }

  virtual std::string name() override { return "TimeStepAssignmentPass"; }
  virtual std::string brief() override {
    return "Assign timestep task for each group.";
  }
};

std::unique_ptr<LgPass> CreateTimeStepAssignmentPass(const LgOptions &options) {
  return std::unique_ptr<LgPass>(new TimeStepAssignmentPass(options));
}

} // namespace tpu
} // namespace tpu_mlir
