#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

struct layer_cycle_info_t {
  int64_t stage;
  int64_t cycle;
  layer_cycle_info_t(int64_t stage, int64_t cycle)
      : stage(stage), cycle(cycle) {}
};

struct gdma_cycle_info_t {
  int64_t stage;
  int64_t cycle;
  int64_t hold_in_lmem; // 1: only load one time 2: tensor has been loaded
  gdma_cycle_info_t(int64_t stage, int64_t cycle, int64_t hold_in_lmem)
      : stage(stage), cycle(cycle), hold_in_lmem(hold_in_lmem) {}
};

void CycleCalculator::set_local_sec_info(local_sec_info_t &sec_info,
                                         Operation *op,
                                         TensorInfo &tensor_infos) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));

  // Note: WhereOp, MaskedFillOp may need to be processed differently.
  int64_t N, C, H, W;
  bool has_input = false;
  Value in = op->getOperand(0);
  auto iter = tensor_infos.find(in);
  if (iter != tensor_infos.end()) {
    module::getNCHW(in, N, C, H, W);
    auto &si = iter->second.slice_info;
    sec_info.n_slice = si.n[0].second;
    sec_info.h_slice = si.h[0].second;
    sec_info.w_slice = W;
    has_input = true;
  }

  Value out = op->getResult(0);
  iter = tensor_infos.find(in);
  if (iter != tensor_infos.end()) {
    module::getNCHW(out, N, C, H, W);
    auto &si = iter->second.slice_info;
    sec_info.out_n_slice = si.n[0].second;
    sec_info.out_h_slice = si.h[0].second;
    sec_info.out_w_slice = W;
    if (!has_input) {
      sec_info.n_slice = si.n[0].second;
      sec_info.h_slice = si.h[0].second;
      sec_info.w_slice = W;
    }
  }
}

int64_t CycleCalculator::getGroupCycle(BasicTimeStepPtr &time_step,
                                       shape_secs_t &shape_secs) {
  int64_t loop_num = shape_secs.nsecs * shape_secs.hsecs;
  std::vector<layer_cycle_info_t> layer_cycle;
  std::vector<gdma_cycle_info_t> gdma_cycle;

  int64_t filling_cycle = 0, kernel_cycle = 0, draining_cycle = 0;
  int64_t total_layer_cycle = 0, total_gdma_cycle = 0;
  int64_t swpipl_stage_num = time_step->get_swpipl_stage_num();
  int64_t timestep_num = time_step->get_timestep_num();
  auto &tensor_infos = time_step->get_tensor_infos();

  for (int64_t ts = 0; ts < timestep_num; ++ts) {
    int64_t start = 0;
    layer_cycle.clear();
    gdma_cycle.clear();
    const TpuTsField &timestep_layers = time_step->getLayers(ts);
    const GdmaTsField &timestep_tensors = time_step->getTensors(ts);
    // record cycle count for all layers and tensors here
    for (auto op : timestep_layers) {
      int64_t stage = time_step->get_layer_swpipl_stage(op);
      int64_t cycle = this->getLocalLayerCycle(op, tensor_infos, false);
      layer_cycle.push_back(layer_cycle_info_t(stage, cycle));
    }
    for (auto tensor : timestep_tensors) {
      int64_t stage = time_step->get_tensor_swpipl_stage(tensor.first);
      int64_t cycle = this->getGdmaCycle(tensor.first, tensor.second);
      int64_t hold_in_lmem =
          time_step->is_tensor_hold_in_lmem(tensor.first) ? 1 : 0;
      gdma_cycle.push_back(gdma_cycle_info_t(stage, cycle, hold_in_lmem));
    }

    // filling time
    for (int64_t j = 0; j < swpipl_stage_num; ++j) {
      start = std::max((j - (loop_num - 1)), (int64_t)0);
      // layers
      total_layer_cycle = 0;
      for (auto &layer : layer_cycle) {
        if (layer.stage <= j && layer.stage >= start) {
          total_layer_cycle += layer.cycle;
        }
      }
      // tensors
      total_gdma_cycle = 0;
      for (auto &tensor : gdma_cycle) {
        if (tensor.stage <= j && tensor.stage >= start &&
            tensor.hold_in_lmem < 2) {
          total_gdma_cycle += tensor.cycle;
          if (tensor.hold_in_lmem == 1) {
            tensor.hold_in_lmem = 2;
          }
        }
      }
      // max
      filling_cycle += std::max(total_layer_cycle, total_gdma_cycle);
    }

    // kernel time
    if (loop_num > swpipl_stage_num) {
      // layers
      total_layer_cycle = 0;
      for (auto &layer : layer_cycle) {
        total_layer_cycle += layer.cycle;
      }
      // tensors
      total_gdma_cycle = 0;
      for (auto &tensor : gdma_cycle) {
        if (tensor.hold_in_lmem == 0) {
          total_gdma_cycle += tensor.cycle;
        }
      }
      kernel_cycle += std::max(total_layer_cycle, total_gdma_cycle);
    }

    // draining time
    for (int64_t j = start + 1; j < swpipl_stage_num; ++j) {
      // layers
      total_layer_cycle = 0;
      for (auto &layer : layer_cycle) {
        if (layer.stage >= j && layer.stage < swpipl_stage_num) {
          total_layer_cycle += layer.cycle;
        }
      }
      // tensors
      total_gdma_cycle = 0;
      for (auto &tensor : gdma_cycle) {
        if (tensor.hold_in_lmem == 0 && tensor.stage >= j &&
            tensor.stage < swpipl_stage_num) {
          total_gdma_cycle += tensor.cycle;
        }
      }
      draining_cycle += std::max(total_layer_cycle, total_gdma_cycle);
    }
  }
  int64_t total_cycle =
      filling_cycle + draining_cycle +
      std::max(loop_num - swpipl_stage_num, (int64_t)0) * kernel_cycle;
  return total_cycle;
}

int64_t Bm168xCycleCalculator::getGlobalLayerCycle(Operation *op) {
  auto bm168x = BM168x::instance();
  bm168x->set_command_issue_flag(false);
  bm168x->reset_cmd_id_node();

  // generate_fake_global_addr(op);
  auto castOp = dyn_cast<GlobalGenInterface>(op);
  castOp.codegen_global_bm168x();

  int64_t cycle = bm168x->get_cmd_cycle();
  bm168x->dl_sg_stas_reset();
  return cycle;
}

int64_t Bm168xCycleCalculator::getLocalLayerCycle(Operation *op,
                                            TensorInfo &tensor_infos,
                                            bool calc_bdc_slack) {
  auto bm168x = BM168x::instance();
  int64_t cycle = 0;
  local_sec_info_t sec_info;
  set_local_sec_info(sec_info, op, tensor_infos);
  auto lgOp = dyn_cast<LocalGenInterface>(op);
  // #pragma omp critical
  {
    bm168x->set_command_issue_flag(false);
    bm168x->reset_cmd_id_node();

    // set_local_layer_io_addr(op);
    lgOp.codegen_local_bm168x(0, 0, sec_info);

    int64_t bdc_cycle = bm168x->get_bdc_cycle();
    int64_t gdma_cycle = bm168x->get_gdma_cycle();
    if (calc_bdc_slack) {
      cycle = bdc_cycle - gdma_cycle;
    } else {
      cycle = bdc_cycle > gdma_cycle ? bdc_cycle : gdma_cycle;
    }
    bm168x->dl_sg_stas_reset();
  }
  return cycle;
}

int64_t Bm168xCycleCalculator::getGdmaCycle(Value v,
                                      const tensor_info_t &tensor_info) {
  auto bm168x = BM168x::instance();
  bm168x->set_command_issue_flag(false);
  bm168x->reset_cmd_id_node();

  // because LoadOp/StoreOp are not created during LayerGroup
  int64_t cycle = 0;
  if (tensor_info.mode == TIMESTEP_LOAD) {
    cycle = getLoadCycle(v, tensor_info);
  } else {
    cycle = getStoreCycle(v, tensor_info);
  }
  bm168x->dl_sg_stas_reset();
  return cycle;
}

int64_t Bm168xCycleCalculator::getLoadCycle(Value v,
                                      const tensor_info_t &tensor_info) {
  // need_info:
  // - n_slice, h_slice, eu_align, g_addr, l_addr
  // - need_bcast, use_3ic
  auto bm168x = BM168x::instance();
  int64_t n_slice, h_slice;
  auto &si = tensor_info.slice_info;
  get_max_slice_nh(si, n_slice, h_slice);
  int64_t use_3ic = tensor_info.use_3ic_opt;
  bool need_bcast = tensor_info.need_bcast;
  bool eu_align = tensor_info.eu_align;
  auto pid_node = (CMD_ID_NODE *)bm168x->dl_create_cmd_id_node();
  bm168x->dl_reset_cmd_id(pid_node);
  auto data_type = BM168x::getDataType(v);
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  module::getNCHW(v, N, C, H, W);

  auto g_stride = bm168x->getGlobalStride(N, C, H, W);
  if (need_bcast) {
    C = Arch::NPU_NUM;
    g_stride.N = 0;
    g_stride.C = 0;
    g_stride.H = 0;
  }
  auto l_stride =
      bm168x->getLocalStride(n_slice, C, h_slice, W, fmt_bytes, eu_align);
  auto g_addr = module::getAddress(v);
  auto l_addr = 0;
  if (use_3ic < 4 && use_3ic > 0) {
    auto use_op = *v.getUsers().begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    auto kernel = module::getI64Array(conv_op.getKernelShape());
    int64_t to_ic =
        use_3ic == 1
            ? kernel->at(0)
            : (use_3ic == 2 ? kernel->at(1) : kernel->at(0) * kernel->at(1));
    for (int i = 0; i < C; ++i) {
      bm168x->dl_tensor_broadcast_move_gen_cmd(
          g_addr + i * W * H * fmt_bytes, 0, l_addr, i * to_ic, n_slice,
          h_slice, W, to_ic, g_stride.N, g_stride.H, l_stride.N, l_stride.H,
          gdma_format, true, GDMA_VALUE_DIR_S2L, pid_node);
    }
  } else {
    bm168x->dl_tensor_stride_move_gen_cmd(
        l_addr, 0, g_addr, n_slice, C, h_slice, W, g_stride.N, g_stride.C,
        g_stride.H, g_stride.W, l_stride.N, l_stride.C, l_stride.H, l_stride.W,
        gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
  }
  int64_t gdma_cycle = bm168x->dl_get_cmd_id_cycle(pid_node);
  bm168x->dl_destroy_cmd_id_node(pid_node);
  return gdma_cycle;
}

int64_t Bm168xCycleCalculator::getStoreCycle(Value v,
                                       const tensor_info_t &tensor_info) {
  // need_info:
  // - n_slice, h_slice, eu_align, g_addr, l_addr
  auto bm168x = BM168x::instance();
  int64_t n_slice, h_slice;
  auto &si = tensor_info.slice_info;
  get_max_slice_nh(si, n_slice, h_slice);
  bool eu_align = tensor_info.eu_align;
  auto pid_node = (CMD_ID_NODE *)bm168x->dl_create_cmd_id_node();
  bm168x->dl_reset_cmd_id(pid_node);
  auto data_type = BM168x::getDataType(v);
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, H, W;
  module::getNCHW(v, N, C, H, W);
  auto g_stride = bm168x->getGlobalStride(N, C, H, W);
  auto l_stride =
      bm168x->getLocalStride(n_slice, C, h_slice, W, fmt_bytes, eu_align);
  auto g_addr = module::getAddress(v);
  int64_t l_addr = 0;

  bm168x->dl_tensor_stride_move_gen_cmd(
      l_addr, 0, g_addr, n_slice, C, h_slice, W, l_stride.N, l_stride.C,
      l_stride.H, l_stride.W, g_stride.N, g_stride.C, g_stride.H, g_stride.W,
      gdma_format, GDMA_VALUE_DIR_L2S, 0, pid_node);

  int64_t gdma_cycle = bm168x->dl_get_cmd_id_cycle(pid_node);
  bm168x->dl_destroy_cmd_id_node(pid_node);
  return gdma_cycle;
}

int64_t Cv18xxCycleCalculator::getGlobalLayerCycle(Operation *op) {
  return 0;
}

int64_t Cv18xxCycleCalculator::getLocalLayerCycle(Operation *op,
                                            TensorInfo &tensor_infos,
                                            bool calc_bdc_slack) {
  return 0;
}

int64_t Cv18xxCycleCalculator::getGdmaCycle(Value v,
                                      const tensor_info_t &tensor_info) {
  return 0;
}

int64_t Cv18xxCycleCalculator::getLoadCycle(Value v,
                                      const tensor_info_t &tensor_info) {
  return 0;
}

int64_t Cv18xxCycleCalculator::getStoreCycle(Value v,
                                       const tensor_info_t &tensor_info) {
  return 0;
}

} // namespace tpu
} // namespace tpu_mlir
