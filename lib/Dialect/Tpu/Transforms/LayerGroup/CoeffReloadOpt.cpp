//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"

namespace tpu_mlir {
namespace tpu {

typedef struct coeff_cost {
  int64_t life_time;
  int64_t gdma_cycle;
} coeff_cost_t;

static bool cmp_by_cost(const std::pair<Value, coeff_cost_t> &lhs,
                        const std::pair<Value, coeff_cost_t> &rhs) {
  bool res = (lhs.second.life_time == rhs.second.life_time)
                 ? (lhs.second.gdma_cycle > rhs.second.gdma_cycle)
                 : (lhs.second.life_time < rhs.second.life_time);
  return res;
}

#define CYCLE_ERROR (0.2)
void coeff_reload_open(BasicTimeStepPtr &time_step, TensorInfo &tensor_infos) {
  std::vector<std::pair<Value, coeff_cost_t>> tensor_to_coeff_cost;
  int64_t cycle_cost, life_time;
  int64_t timestep_num = time_step->get_timestep_num();
  std::unique_ptr<CycleCalculator> cycle_calculator;
  if (module::isCV18xx()) {
    Cv18xxCycleCalculator *cyc_ptr = new Cv18xxCycleCalculator();
    cycle_calculator.reset(cyc_ptr);
  } else {
    Bm168xCycleCalculator *cyc_ptr = new Bm168xCycleCalculator();
    cycle_calculator.reset(cyc_ptr);
  }
  time_step->gen_all_mem_buffer_ts();
  for (int64_t ts = 0; ts < timestep_num; ++ts) {
    int64_t slack = 0;
    tensor_to_coeff_cost.clear();
    const GdmaTsField &ts_tensors = time_step->getTensors(ts);
    if (ts_tensors.empty()) {
      continue;
    }
    for (auto &tensor : ts_tensors) {
      tensor_info_t &tensor_info = tensor_infos[tensor.first];
      cycle_cost = cycle_calculator->getGdmaCycle(tensor.first, tensor_info,
                                                  GROUP_NORMAL);
      if (time_step->is_tensor_hold_in_lmem(tensor.first)) {
        life_time = time_step->get_tensor_life_time(tensor.first);
        coeff_cost_t cost = {life_time, cycle_cost};
        tensor_to_coeff_cost.push_back(std::make_pair(tensor.first, cost));
      } else {
        slack -= cycle_cost;
      }
    }
    if (tensor_to_coeff_cost.empty()) {
      continue;
    }

    const TpuTsField &ts_layers = time_step->getLayers(ts);
    for (auto op : ts_layers) {
      cycle_cost = cycle_calculator->getLocalLayerCycle(op, tensor_infos,
                                                        GROUP_NORMAL, true);
      slack += cycle_cost;
    }
    if (slack < 0) {
      continue;
    }

    // consider the cycle error rate
    slack = (int64_t)(slack * (1.0f - CYCLE_ERROR));
    // schedule some coeffs to be reloaded
    std::sort(tensor_to_coeff_cost.begin(), tensor_to_coeff_cost.end(),
              cmp_by_cost);
    for (size_t i = 0; i < tensor_to_coeff_cost.size(); ++i) {
      cycle_cost = tensor_to_coeff_cost[i].second.gdma_cycle;
      if (cycle_cost < slack) {
        time_step->cancel_tensor_hold_in_lmem(tensor_to_coeff_cost[i].first);
        slack -= cycle_cost;
      } else {
        break;
      }
    }
  }
}

} // namespace tpu
} // namespace tpu_mlir
