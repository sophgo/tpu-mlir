//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/CycleCalculator.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1688.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_local_api.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx_profiling.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/CoreParallel/CoreParallel.hpp"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>
#include "tpu_mlir/Interfaces/InplaceInterface.h"
#include "tpu_mlir/Interfaces/InplaceInterface.cpp.inc"


#define DEBUG_TYPE "layer-group"

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
                                         TensorInfo &tensor_infos,
                                         group_type_t group_type) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  // Note: WhereOp, MaskedFillOp may need to be processed differently.
  // int64_t N, C, D, H, W;
  bool has_input = false;
  Value in = op->getOperand(0);
  auto iter = tensor_infos.find(in);
  if (iter != tensor_infos.end()) {
    // module::getNCDHW(in, N, C, D, H, W, group_type);
    auto &si = iter->second.slice_info;
    sec_info.n_slice = si.n[0].second;
    sec_info.h_slice = si.h[0].second;
    sec_info.d_slice = si.d[0].second;
    sec_info.w_slice = si.w[0].second;
    sec_info.c_slice = si.c[0].second;
    has_input = true;
  }

  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp, tpu::DivOp, tpu::MaxOp,
          tpu::MinOp>(op)) {
    Value in2 = op->getOperand(1);
    auto iter = tensor_infos.find(in2);
    if (iter != tensor_infos.end()) {
      // module::getNCDHW(in2, N, C, D, H, W, group_type);
      auto &si = iter->second.slice_info;
      sec_info.n_slice = std::max(si.n[0].second, (int64_t)sec_info.n_slice);
      sec_info.h_slice = std::max(si.h[0].second, (int64_t)sec_info.h_slice);
      sec_info.d_slice = std::max(si.d[0].second, (int64_t)sec_info.d_slice);
      sec_info.w_slice = std::max(si.w[0].second, (int64_t)sec_info.w_slice);
      sec_info.c_slice = std::max(si.c[0].second, (int64_t)sec_info.c_slice);
    }
  }

  Value out = op->getResult(0);
  iter = tensor_infos.find(out);
  if (iter != tensor_infos.end()) {
    // module::getNCDHW(out, N, C, D, H, W, group_type);
    auto &si = iter->second.slice_info;
    sec_info.out_n_slice = si.n[0].second;
    sec_info.out_h_slice = si.h[0].second;
    // sec_info.out_d_slice = si.d[0].second;
    sec_info.out_w_slice = si.w[0].second;
    if (!has_input) {
      sec_info.n_slice = si.n[0].second;
      sec_info.h_slice = si.h[0].second;
      sec_info.d_slice = si.d[0].second;
      sec_info.w_slice = si.w[0].second;
      sec_info.c_slice = si.c[0].second;
    }
  }
}


SmallVector<Operation *> createTempCoreParallelOp(Operation *_op) {
  SmallVector<Operation *> computeOps;
  if (!isa<IndexingMapsInterface>(_op)) {
    return computeOps;
  }

  auto gl = dyn_cast<GlobalGenInterface>(_op);
  if (gl && gl.support_multi_core()) {
    return computeOps;
  }

  auto op = dyn_cast<IndexingMapsInterface>(_op);
  // int offset = 0;
  int num_core = module::getCoreNum();

  LLVM_DEBUG({
    llvm::errs() << "IndexingMap" << '\n';
    op.dump();
  });

  if (getRunMode(op) == RunMode::TPU_DYNAMIC)
    return computeOps;
  if (num_core < 2)
    return computeOps;
  auto indexMap = op.getIndexingMaps();
  if (!indexMap || indexMap.empty())
    return computeOps;

  auto operandsMap = indexMap.getValue().slice(0, op->getNumOperands());
  auto resultsMap =
      indexMap.getValue().slice(op->getNumOperands(), op->getNumResults());

  // use the first resultsMap as a sample, each AffineMap in indexingMap have
  // the same dimCount.
  auto resultMap = cast<AffineMapAttr>(resultsMap[0]).getValue();
  if (!resultMap.isIdentity() || resultMap.isEmpty())
    return computeOps;

  // :load balance:
  // shape = [a, b]; other situations can be reduced to this formula.
  // a * b = a * \sum_{i=1}^n (b_i)
  // a * n <= num_core
  // Find the largest n
  // This implement use the maxSlice as much as possible, but it does not take
  // the number of NPU into account. #please improve This
  auto shapeParallel = SmallVector<int64_t>(
      module::getShape(op->getResult(0)).slice(0, resultMap.getNumInputs()));

  int splitDim = 0, splitMax = 1;
  SmallVector<int64_t, 4> iterationShape;

  { // This is a temporary fix for GroupNorm support; Try to refactor this out.
    if (auto groupNormOp = dyn_cast<tpu::GroupNormOp>(op.getOperation())) {
      shapeParallel[1] = groupNormOp.getNumGroups();
    }
  }

  for (int64_t i = 0, n = shapeParallel.size(), iterSpace = 1; i < n; ++i) {
    if (iterSpace * shapeParallel[i] >= num_core) {
      splitDim = i;
      int coreK = num_core / iterSpace;                  // This is the lower n
      splitMax = (shapeParallel[i] + coreK - 1) / coreK; // This is max(b_i)
      auto n = (shapeParallel[i] + splitMax - 1) / splitMax;
      iterationShape.push_back(n);
      break;
    } else {
      iterationShape.push_back(shapeParallel[i]);
      iterSpace *= shapeParallel[i];
    }
  }

  if (splitDim == 0 && iterationShape[0] == 1)
    return computeOps;

  if (auto matmulOp = dyn_cast<tpu::MatMulOp>(op.getOperation())) {
    auto l_shape = module::getShape(matmulOp.getInput());
    auto r_shape = module::getShape(matmulOp.getRight());

    if (l_shape.size() == 4 && r_shape.size() == 4 &&
        l_shape[0] == r_shape[0] && l_shape[1] != r_shape[1] &&
        l_shape[2] == r_shape[2] && l_shape[3] == r_shape[3])
      return computeOps;
  }

  { // This is a temporary fix for GroupNorm support; Try to refactor this out.
    if (auto groupNormOp = dyn_cast<tpu::GroupNormOp>(op.getOperation())) {
      auto channel = module::getShape(groupNormOp.getInput())[1];
      shapeParallel[1] = channel;
      if (splitDim == 1)
        splitMax *= channel / groupNormOp.getNumGroups();
    }
  }

  // auto rewriter = IRRewriter(op.getContext());
  // rewriter.setInsertionPoint(op);
  auto builder = OpBuilder(op.getContext());

  // rewriter.setInsertionPointToStart(body);
  // rewriter.replaceAllUsesWith(op->getResults(), parallelOp.getResults());

  // Travel the multi-dimensional iteration space.
  // 1. build split operation for each operand.
  SmallVector<Operation *, 4> splitOps;
  SmallVector<SmallVector<int64_t, 4>, 4> operandsStride;
  for (auto [index, valueMap, value] :
       llvm::enumerate(operandsMap, op->getOperands())) {
    if (auto outTypes = getSplitTypes(valueMap, value, ArrayRef(shapeParallel),
                                      splitDim, splitMax)) {
      auto name = module::getName(value) + "_" + Twine(index).str();
      auto nameLoc = NameLoc::get(builder.getStringAttr(name));

      splitOps.push_back(builder.create<tpu::SplitOp>(
          nameLoc, TypeRange(outTypes.value()), value));
    } else {
      splitOps.push_back(value.getDefiningOp());
    }
    operandsStride.push_back(
        getValidStride(valueMap, ArrayRef(iterationShape)));
  }

  // 2. build distributing compute operation for each num_core.

  SmallVector<SmallVector<Type>> outputsTypes;
  for (auto [valueMap, value] : llvm::zip(resultsMap, op->getResults())) {
    outputsTypes.push_back(getSplitTypes(valueMap, value,
                                         ArrayRef(shapeParallel), splitDim,
                                         splitMax)
                               .value());
  }

  auto resultStride = getValidStride(resultsMap[0], ArrayRef(iterationShape));

  auto createComputeOp = [&](ArrayRef<int64_t> dims) {
    SmallVector<Value, 4> operands;
    for (auto [index, spOp] : llvm::enumerate(splitOps)) {
      ArrayRef stride(operandsStride[index]);
      if (spOp)
        operands.push_back(spOp->getResult(getValidIndex(dims, stride)));
      else // inputs
        operands.push_back(op->getOperand(index));
    }

    SmallVector<Type, 2> resultsType;
    for (auto types : outputsTypes)
      resultsType.push_back(types[getValidIndex(dims, ArrayRef(resultStride))]);

    auto suffix =
        llvm::formatv("_{0:$[_]}", make_range(dims.begin(), dims.end()));
    auto name = module::getName(op, 0) + suffix.str().c_str();
    auto nameLoc = NameLoc::get(builder.getStringAttr(name));

    computeOps.push_back(builder.create(nameLoc, op->getName().getIdentifier(),
                                        operands, resultsType, op->getAttrs()));
    { // This is a temporary fix for GroupNorm support; Try to refactor this
      // out.
      if (auto groupNormOp = dyn_cast<tpu::GroupNormOp>(computeOps.back())) {
        auto numGroup = groupNormOp.getNumGroups();
        auto itemPerGroup = module::getShape(op->getOperand(0))[1] / numGroup;
        auto channel = module::getShape(groupNormOp.getInput())[1];
        groupNormOp.setNumGroups(channel / itemPerGroup);
      }
    }
  };

  invokeInIterationSpace(ArrayRef(iterationShape), createComputeOp);

  return computeOps;
}

void removeTempCoreParallelOp(SmallVector<Operation *> ops) {
  std::set<Operation *> splits;
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (auto splitOp = dyn_cast_or_null<tpu::SplitOp>(v.getDefiningOp())) {
        splits.insert(splitOp.getOperation());
      }
    }
  }

  for (auto op : splits) {
    auto rewriter = IRRewriter(op->getContext());
    for (auto user : op->getUsers()) {
      // user->dump();
      rewriter.eraseOp(user);
    }
    rewriter.eraseOp(op);
  }
}

int64_t CycleCalculator::getGroupCycle(BasicTimeStepPtr &time_step,
                                       shape_secs_t &shape_secs,
                                       group_type_t group_type) {
  // (void*)invokeInIterationSpace;

  int64_t loop_num =
      shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs * shape_secs.dsecs * shape_secs.wsecs;


  DEBUG_WITH_TYPE("cycle_calc", {
    llvm::dbgs() << "; action = cycle_calc"
    << "; loop_num = " << loop_num
    << "\n";
  });
  std::vector<layer_cycle_info_t> layer_cycle;
  std::vector<gdma_cycle_info_t> gdma_cycle;
  bool consider_multi_core_bw = false;
  auto num_core = module::getCoreNum();
  if (num_core == 2) {
    consider_multi_core_bw = true;
    loop_num = loop_num / num_core + (loop_num % num_core > 0);
    DEBUG_WITH_TYPE("cycle_calc", {
    llvm::dbgs() << "; action = cycle_calc"
                  << "; step = multi_core_refactor"
                  << "; loop_num = " << loop_num << "\n";
  });
  }

    if(num_core == 8){
      loop_num = loop_num / num_core + (loop_num % num_core > 0);
          DEBUG_WITH_TYPE("cycle_calc", {
        llvm::dbgs() << "; action = cycle_calc"
          << "; step = multi_core_refactor"
          << "; loop_num = " << loop_num << "\n";
      });
    }


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
      int64_t cycle =
          this->getLocalLayerCycle(op, tensor_infos, group_type, false);
      DEBUG_WITH_TYPE("cycle_calc", {
        llvm::dbgs() << "; action = cycle_calc"
                     << "; engine = layer_cycle"
                     << "; op_name = " << module::getName(op)
                     << "; value = " << cycle << "\n";
        op->dump();
      });
      layer_cycle.push_back(layer_cycle_info_t(stage, cycle));
    }
    for (auto tensor : timestep_tensors) {
      int64_t stage = time_step->get_tensor_swpipl_stage(tensor.first);
      auto tensor_info = tensor.second;
      if (module::getChip() == module::Chip::BM1688) {
        auto bm1688 = (BM1688 *)BM168x::instance();
        float BW = 24;
        if (consider_multi_core_bw) {
          BW = 15.f;
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = multi_core_align"
                         << "; BW = " << BW
                         << "\n";
          });
          bm1688->dl_set_gdma_bw_s2l(BW);
          bm1688->dl_set_gdma_bw_l2s(BW);
        }
      }
      int64_t cycle = this->getGdmaCycle(tensor.first, tensor_info, group_type);
      if (module::getChip() == module::Chip::BM1688) {
        auto bm1688 = (BM1688 *)BM168x::instance();
        float BW = 24;
        bm1688->dl_set_gdma_bw_s2l(BW);
        bm1688->dl_set_gdma_bw_l2s(BW);
      }
      int64_t hold_in_lmem =
          time_step->is_tensor_hold_in_lmem(tensor.first) ? 1 : 0;
      DEBUG_WITH_TYPE("cycle_calc", {
        llvm::dbgs() << "; action = cycle_calc"
                     << "; engine = gdma_cycle"
                     << "; op_name = " << module::getName(tensor.first)
                     << "; value = " << cycle << "\n";
        tensor.first.dump();
      });
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
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = accumulate_tiu_cycle"
                         << "; stage = filling"
                         << "; value = " << layer.cycle
                         << "\n";
          });
        }
      }
      // tensors
      total_gdma_cycle = 0;
      for (auto &tensor : gdma_cycle) {
        if (tensor.stage <= j && tensor.stage >= start &&
            tensor.hold_in_lmem < 2) {
          total_gdma_cycle += tensor.cycle;
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = accumulate_gdma_cycle"
                         << "; stage = filling"
                         << "; value = " << tensor.cycle
                         << "\n";
          });
          if (tensor.hold_in_lmem == 1) {
            tensor.hold_in_lmem = 2;
          }
        }
      }
      DEBUG_WITH_TYPE("cycle_calc", {
        if (total_layer_cycle > total_gdma_cycle) {
          llvm::dbgs() << "; action = consider_tiu_cycle"
                       << "; stage = filling"
                       << "; value = " << total_layer_cycle
                       << "\n";
        } else {
          llvm::dbgs() << "; action = consider_gdma_cycle"
                       << "; stage = filling"
                       << "; value = " << total_gdma_cycle
                       << "\n";
        }
      });
      // max
      filling_cycle += std::max(total_layer_cycle, total_gdma_cycle);
    }
    // kernel time
    if (loop_num > swpipl_stage_num) {
      // layers
      total_layer_cycle = 0;
      for (auto &layer : layer_cycle) {
        total_layer_cycle += layer.cycle;
        DEBUG_WITH_TYPE("cycle_calc", {
          llvm::dbgs() << "; action = accumulate_tiu_cycle"
                       << "; stage = kernel"
                       << "; value = " << layer.cycle
                       << "\n";
        });
      }
      // tensors
      total_gdma_cycle = 0;
      for (auto &tensor : gdma_cycle) {
        if (tensor.hold_in_lmem == 0) {
          total_gdma_cycle += tensor.cycle;
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = accumulate_gdma_cycle"
                         << "; stage = kernel"
                         << "; value = " << tensor.cycle
                         << "\n";
          });
        }
      }
      DEBUG_WITH_TYPE("cycle_calc", {
        if (total_layer_cycle > total_gdma_cycle) {
          llvm::dbgs() << "; action = consider_tiu_cycle"
                       << "; stage = kernel"
                       << "; value = " << total_layer_cycle << "\n";
        } else {
          llvm::dbgs() << "; action = consider_gdma_cycle"
                       << "; stage = kernel"
                       << "; value = " << total_gdma_cycle << "\n";
        }
      });
      kernel_cycle += std::max(total_layer_cycle, total_gdma_cycle);
    }
    // draining time
    for (int64_t j = start + 1; j < swpipl_stage_num; ++j) {
      // layers
      total_layer_cycle = 0;
      for (auto &layer : layer_cycle) {
        if (layer.stage >= j && layer.stage < swpipl_stage_num) {
          total_layer_cycle += layer.cycle;
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = accumulate_tiu_cycle"
                         << "; stage = draining"
                         << "; value = " << layer.cycle
                         << "\n";
          });
        }
      }
      // tensors
      total_gdma_cycle = 0;
      for (auto &tensor : gdma_cycle) {
        if (tensor.hold_in_lmem == 0 && tensor.stage >= j &&
            tensor.stage < swpipl_stage_num) {
          total_gdma_cycle += tensor.cycle;
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = accumulate_gdma_cycle"
                         << "; stage = draining"
                         << "; value = " << tensor.cycle
                         << "\n";
          });
        }
      }
      DEBUG_WITH_TYPE("cycle_calc", {
        if (total_layer_cycle > total_gdma_cycle) {
          llvm::dbgs() << "; action = consider_tiu_cycle"
                       << "; stage = draining"
                       << "; value = " << total_layer_cycle << "\n";
        } else {
          llvm::dbgs() << "; action = consider_gdma_cycle"
                       << "; stage = draining"
                       << "; value = " << total_gdma_cycle << "\n";
        }
      });
      draining_cycle += std::max(total_layer_cycle, total_gdma_cycle);
    }
  }
  int64_t total_cycle =
      filling_cycle + draining_cycle +
      std::max(loop_num - swpipl_stage_num, (int64_t)0) * kernel_cycle;
  DEBUG_WITH_TYPE("cycle_calc", {
    llvm::dbgs() << "; action = total_cycle_of_group"
                 << "; step = end"
                 << "; event = " << "total_cycle_of_group"
                 << "; filling_cycle = " << filling_cycle
                 << "; draining_cycle = " << draining_cycle
                 << "; loop_num = " << loop_num
                 << "; swpipl_stage_num = " << swpipl_stage_num
                 << "; kernel_cycle = " << kernel_cycle
                 << "; total_cycle = " << total_cycle << "\n";
  });
  return total_cycle;
}

int64_t Bm168xCycleCalculator::getGlobalLayerCycle(Operation *op) {
  auto splitedOps = createTempCoreParallelOp(op);
  // SmallVector<Operation *> splitedOps;

  if (auto inplaceOp = dyn_cast<InplaceInterface>(op)) {
      if(inplaceOp.supportInplace()){
        return 0;
      }
  }

  if (splitedOps.size() == 0) {
    splitedOps.push_back(op);
  }

  auto bm168x = BM168x::instance();

  if (module::getChip() == module::Chip::BM1688) {
    auto bm1688 = (BM1688 *)bm168x;
    bool imp_multi_core_global = isa<tpu::Conv2DOp>(op) || splitedOps.size() > 1;
    float BW = 24;
    if (imp_multi_core_global) {
      BW = 15.f;
      DEBUG_WITH_TYPE("cycle_calc", {
        llvm::dbgs() << "; action = align_multi_core_bw"
                     << "; BW = " << BW << "; op_name = " << module::getName(op)
                     << "\n";
      });
    }
    // for other targets, writer
    bm1688->dl_set_gdma_bw_s2l(BW);
    bm1688->dl_set_gdma_bw_l2s(BW);
  }
  int64_t cycle = 0;

  for (auto _op : splitedOps) {
    if (auto castOp = dyn_cast<GlobalGenInterface>(_op)) {
      bm168x->set_command_issue_flag(false);
      bm168x->reset_cmd_id_node();
      castOp.codegen_global_bm168x();
      cycle = std::max(cycle, bm168x->get_cmd_cycle());
      bm168x->dl_sg_stas_reset();
    }
  }

  DEBUG_WITH_TYPE("mc_lg_refactor", {
    // for multicore implements
    if (auto castOp = dyn_cast<GlobalGenInterface>(op)) {
      bm168x->set_command_issue_flag(false);
      bm168x->reset_cmd_id_node();
      castOp.codegen_global_bm168x();
      auto full_cycle = bm168x->get_cmd_cycle();
      DEBUG_WITH_TYPE("cycle_calc", {
        llvm::dbgs() << "; action = compare_global_and_core_parallel"
                      << "; op_name = " << module::getName(op)
                      << "; full = " << full_cycle
                      << "; core_parallel = " << cycle
                      << "\n";
      });
      cycle = std::min(cycle, full_cycle);
      bm168x->dl_sg_stas_reset();
    }
  });

  if (splitedOps.size() > 1) {
    removeTempCoreParallelOp(splitedOps);
  }

  // restore default BW
  if (module::getChip() == module::Chip::BM1688) {
    auto bm1688 = (BM1688 *)bm168x;
    float BW = 24;
    // for other targets, writer
    bm1688->dl_set_gdma_bw_s2l(BW);
    bm1688->dl_set_gdma_bw_l2s(BW);
  }
  return cycle;
}

int64_t Bm168xCycleCalculator::getLocalLayerCycle(Operation *op,
                                                  TensorInfo &tensor_infos,
                                                  group_type_t group_type,
                                                  bool calc_bdc_slack) {
  auto bm168x = BM168x::instance();
  int64_t cycle = 0;
  local_sec_info_t sec_info;
  set_local_sec_info(sec_info, op, tensor_infos, group_type);
  auto lgOp = dyn_cast<LocalGenInterface>(op);
  // #pragma omp critical
  {
    bm168x->set_command_issue_flag(false);
    bm168x->reset_cmd_id_node();
    DEBUG_WITH_TYPE("cycle_calc_cmd", {
      llvm::dbgs() << "; action = codegen_local_layer"
                   << "; op_name = " << module::getName(op)
                   << "; tiu_dma_id(before) = " << ((int*)((*BM168x::instance())->bdc_node))[1]
                   << "\n";
    });
    // set_local_layer_io_addr(op);
    lgOp.codegen_local_bm168x(0, 0, 0, 0, 0, group_type, sec_info);

    DEBUG_WITH_TYPE("cycle_calc_cmd", {
      llvm::dbgs() << "; action = codegen_local_layer"
                   << "; op_name = " << module::getName(op)
                   << "; tiu_dma_id(after) = " << bm168x->get_total_id("tiu:0:0")
                   << "; tiu_dma_id(before) = " << ((int*)((*BM168x::instance())->bdc_node))[1]
                   << "\n";
    });
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
                                            tensor_info_t &tensor_info,
                                            group_type_t group_type, Operation* owner_op, int mode) {
  auto bm168x = BM168x::instance();
  bm168x->set_command_issue_flag(false);
  bm168x->reset_cmd_id_node();

  // because LoadOp/StoreOp are not created during LayerGroup
  int64_t cycle = 0;
  if (tensor_info.mode2 > 0) {
    if (tensor_info.mode2 & TIMESTEP2_LOAD) {
      cycle = getLoadCycle(v, tensor_info, group_type, owner_op);
    } else if (tensor_info.mode2 & TIMESTEP2_STORE) {
      cycle = getStoreCycle(v, tensor_info, group_type);
    } else if (tensor_info.mode2 & TIMESTEP2_STORE_AND_LOAD) {
      if (mode == 0) {
        cycle = getStoreCycle(v, tensor_info, group_type);
      } else if (mode == 1) {
        cycle = getLoadCycle(v, tensor_info, group_type, owner_op);
      }
    }
  } else {
    if (tensor_info.mode == TIMESTEP_LOAD) {
      cycle = getLoadCycle(v, tensor_info, group_type);
    } else {
      cycle = getStoreCycle(v, tensor_info, group_type);
    }
  }
  bm168x->dl_sg_stas_reset();
  return cycle;
}


int64_t Bm168xCycleCalculator::getLoadCycle(Value v,
                                            tensor_info_t &tensor_info,
                                            group_type_t group_type, Operation* owner_op) {
  // need_info:
  // - n_slice, h_slice, eu_align, g_addr, l_addr
  // - need_bcast, use_3ic
  // TODO: CONCAT
  auto bm168x = BM168x::instance();
  int64_t n_slice, c_slice, h_slice, d_slice, w_slice;
  auto si = tensor_info.slice_info;
  if (owner_op) {
    si = tensor_info.slice_infos[owner_op];
  }
  get_max_slice_nchdw(si, n_slice, c_slice, h_slice, d_slice, w_slice);
  std::vector<slice_pair_t> slice_idx = get_max_slice_nchdw_and_idx(si, n_slice, c_slice, h_slice, d_slice, w_slice);
  int64_t use_3ic = tensor_info.use_3ic_opt;
  bool need_bcast = tensor_info.need_bcast;
  bool eu_align = tensor_info.eu_align;
  auto pid_node = (CMD_ID_NODE *)bm168x->dl_create_cmd_id_node();
  bm168x->dl_reset_cmd_id(pid_node);
  auto data_type = BM168x::getDataType(v);
  int64_t gdma_format;
  int64_t N, C, D, H, W;
  module::getNCDHW(v, N, C, D, H, W, group_type);
  if (data_type == DTYPE_UINT4 || data_type == DTYPE_INT4) {
    gdma_format = BM168x::GDMA_VALUE_FORMAT_INT8;
    data_type = DTYPE_INT8;
    W >>= 1;
  }
  gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  auto g_addr = module::getAddress(v);
  auto l_addr = 0;
  if (module::isBM1684Family() && fmt_bytes == 1 && use_3ic) {
    if (need_bcast) {
      C = BM168x::NPU_NUM;
    }
    int64_t n_idx = slice_idx[0].second;
    auto use_op = *v.getUsers().begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    auto conv_param = conv_op.parseParam();
    auto kernel = module::getI64Array(conv_op.getKernelShape());
    int data_len = 4;
    int n_idx_trans = n_idx / 4;
    int n_slice_trans = ceiling_func(n_slice, 4);
    int oh_slice = (h_slice - conv_param.kh) / conv_param.sh + 1;
    auto local_offerset = get_buffer_size(v, tensor_info, group_type, owner_op);
    for (int i = 0; i < n_slice_trans; i++) {
      int src_N = C; // ic
      int src_C = conv_param.kh;
      int src_H = oh_slice;
      int src_W = w_slice; // w

      int src_N_stride = H * W;
      int src_C_stride = W;
      int src_H_stride = W * conv_param.sh;
      int src_W_stride = 1;

      int dst_N = 1;
      int dst_C = C * conv_param.kh;
      int dst_H = oh_slice;
      int dst_W = w_slice;

      int dst_W_stride = 1;
      int dst_H_stride = dst_W;

      auto NPU_NUM = BM168x::NPU_NUM;
      auto EU_NUM = BM168x::eu_num(sizeof(float));
      int dst_C_stride = (oh_slice * dst_W + EU_NUM - 1) / EU_NUM * EU_NUM;
      int dst_N_stride =
          (C * conv_param.kh + NPU_NUM - 1) / NPU_NUM * dst_C_stride;
      BM1684::instance().dl_tensor_general_move_gen_cmd(
          g_addr + (uint64_t)(n_idx_trans + i) * C * H * W * data_len +
              (uint64_t)(slice_idx[2].second) * W * data_len + slice_idx[4].second * data_len,
          0, // no use
          src_N, src_C, src_H, src_W, src_N_stride, src_C_stride, src_H_stride,
          src_W_stride, 0, local_offerset + i * dst_N_stride * data_len, 0, dst_N,
          dst_C, dst_H, dst_W, dst_N_stride, dst_C_stride, dst_H_stride,
          dst_W_stride, 0,
          GDMA_VALUE_DIR_S2L, // 0
          0, pid_node);
    }
  } else {
    if (use_3ic < 4 && use_3ic > 0) {
      // correspoding to NEURON_3IC
      auto g_stride = bm168x->getGlobalStride(N, C, H, W);
      if (need_bcast) {
        C = Arch::NPU_NUM;
        g_stride.N = 0;
        g_stride.C = 0;
        g_stride.H = 0;
      }
      auto l_stride = bm168x->getLocalStride(n_slice, C, h_slice, w_slice,
                                             fmt_bytes, eu_align);
      auto use_op = *v.getUsers().begin();
      auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
      auto kernel = module::getI64Array(conv_op.getKernelShape());
      int64_t to_ic =
          use_3ic == 1
              ? kernel->at(0)
              : (use_3ic == 2 ? kernel->at(1) : kernel->at(0) * kernel->at(1));
      for (int64_t i = 0; i < C; ++i) {
        bm168x->dl_tensor_broadcast_move_gen_cmd(
            g_addr + i * W * H * fmt_bytes, 0, l_addr, i * to_ic, n_slice,
            h_slice, w_slice, to_ic, g_stride.N, g_stride.H, l_stride.N,
            l_stride.H, gdma_format, true, GDMA_VALUE_DIR_S2L, pid_node);
      }
    } else {
      // correspoding to NEURON
      int64_t c_num_local = ceiling_func(C, Arch::NPU_NUM);
      int64_t c_stride =
          eu_align ? align_up(h_slice * w_slice, Arch::eu_num(fmt_bytes))
                   : h_slice * w_slice;
      int64_t channel_num = c_slice;
      const int64_t csecs = ceiling_func(channel_num, (int64_t)MAX_TPU_DIM);
      if (d_slice <= n_slice) {
        for (int64_t d = 0; d < d_slice; d++) {
          int64_t channel_index = 0;
          while (channel_index < csecs) {
            int64_t cur_cslice =
                std::min(channel_num - channel_index * (int64_t)MAX_TPU_DIM,
                         (int64_t)MAX_TPU_DIM);
            bm168x->dl_tensor_stride_move_gen_cmd(
                l_addr, 0, g_addr, // only simulate for calc cycle
                n_slice, cur_cslice, h_slice, w_slice, C * D * H * W, D * H * W,
                W, 1, c_num_local * c_stride, c_stride, w_slice, 1, gdma_format,
                GDMA_VALUE_DIR_S2L, 0, pid_node);
            channel_index++;
          }
        }      // depth loop
      } else { // HAVE DEPTH,3D [N,C,D,H,W]->[d,n_slice,c,h_slice,w]
        for (int64_t i = 0; i < n_slice; i++) {
          bm168x->dl_tensor_stride_move_gen_cmd(
              l_addr, 0, g_addr, d_slice, c_slice, h_slice, w_slice,
              H * W,     // actually global d_stride
              D * H * W, // actually global c_stride
              W, 1,
              n_slice * c_num_local * c_stride, // actually local d_stride
              c_stride, w_slice, 1, gdma_format, GDMA_VALUE_DIR_S2L, 0,
              pid_node);
        } // nslice loop
      }
    }
  }
  int64_t gdma_cycle = bm168x->dl_get_cmd_id_cycle(pid_node);
  bm168x->dl_destroy_cmd_id_node(pid_node);
  return gdma_cycle;
}

int64_t Bm168xCycleCalculator::getStoreCycle(Value v,
                                             const tensor_info_t &tensor_info,
                                             group_type_t group_type) {
  // need_info:
  // - n_slice, h_slice, eu_align, g_addr, l_addr
  // TODO: CONCAT BMNET_REORG
  auto bm168x = BM168x::instance();
  int64_t n_slice, c_slice, h_slice, d_slice, w_slice;
  auto &si = tensor_info.slice_info;
  get_max_slice_nchdw(si, n_slice, c_slice, h_slice, d_slice, w_slice);
  auto pid_node = (CMD_ID_NODE *)bm168x->dl_create_cmd_id_node();
  bm168x->dl_reset_cmd_id(pid_node);
  auto data_type = BM168x::getDataType(v);
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t N, C, D, H, W;
  module::getNCDHW(v, N, C, D, H, W, group_type);
  auto g_addr = module::getAddress(v);
  int64_t l_addr = 0;

  int64_t c_num_local = ceiling_func(c_slice, Arch::NPU_NUM);
  int64_t c_stride = align_up(h_slice * w_slice, Arch::eu_num(fmt_bytes));
  int64_t channel_num = c_slice;

  if (d_slice <= n_slice) {
    const int64_t csecs = ceiling_func(channel_num, (int64_t)MAX_TPU_DIM);
    for (int64_t d = 0; d < d_slice; d++) {
      int64_t channel_index = 0;
      while (channel_index < csecs) {
        int64_t cur_cslice =
            std::min(channel_num - channel_index * (int64_t)MAX_TPU_DIM,
                     (int64_t)MAX_TPU_DIM);
        bm168x->dl_tensor_stride_move_gen_cmd(
            l_addr, 0, g_addr, n_slice, cur_cslice, h_slice, w_slice,
            c_num_local * c_stride, c_stride, w_slice, 1, C * D * H * W,
            D * H * W, W, 1, gdma_format,
            GDMA_VALUE_DIR_L2S, // 1,
            0, pid_node);
        channel_index++;
      }
    }
  } else { // HAVE DEPTH,3D [D,n_slice,C,h_slice,W] -> [N,C,D,H,W]
    for (int64_t i = 0; i < n_slice; i++) {
      bm168x->dl_tensor_stride_move_gen_cmd(
          l_addr, 0, g_addr, d_slice, c_slice, h_slice, w_slice,
          n_slice * c_num_local * c_stride, c_stride, w_slice, 1, H * W,
          D * H * W, W, 1, gdma_format,
          GDMA_VALUE_DIR_L2S, // 1,
          0, pid_node);
    }
  }

  int64_t gdma_cycle = bm168x->dl_get_cmd_id_cycle(pid_node);
  bm168x->dl_destroy_cmd_id_node(pid_node);
  return gdma_cycle;
}

int64_t Cv18xxCycleCalculator::getGlobalLayerCycle(Operation *op) {
  std::vector<uint8_t> cmdbuf;
  auto castOp = dyn_cast<GlobalGenInterface>(op);
  castOp.codegen_global_cv18xx(0);
  CV18xx::submit();
  CV18xx::read_cmdbuf(cmdbuf);
  uint64_t cycle = CV18xxProfiling::get_cycle(cmdbuf);
  // {
  //   static int count = 0;
  //   std::stringstream ss;
  //   ss << "cmdbuf_" << count++ << ".bin";
  //   std::ofstream ofs(ss.str(), std::ios::binary);
  //   ofs.write((char *)cmdbuf.data(), cmdbuf.size());
  // }
  return cycle;
}

bool Cv18xxCycleCalculator::check_lmem(Operation *op,
                                       const TensorInfo &tensor_infos,
                                       group_type_t group_type) {
  // simply check if local memory is enough
  int64_t total_size = 0;
  auto ins = get_input_values(op);
  auto outs = get_output_values(op);
  for (auto in : ins) {
    auto iter = tensor_infos.find(in);
    if (iter == tensor_infos.end())
      continue;
    auto &si = iter->second.slice_info;
    if (!module::isWeight(in)) {
      if (si.h[0].second > (4095-32)) {
        return false;
      }
      total_size += Arch::get_tensor_lmem_bytes(
          in, si.n[0].second, si.c[0].second, si.d[0].second, si.h[0].second,
          si.w[0].second);
    }
  }
  for (auto out : outs) {
    auto iter = tensor_infos.find(out);
    if (iter == tensor_infos.end())
      continue;
    auto &si = iter->second.slice_info;
    if (si.h[0].second > (4095-32)) {
      return false;
    }
    total_size += Arch::get_tensor_lmem_bytes(out, si.n[0].second,
                                              si.c[0].second, si.d[0].second,
                                              si.h[0].second, si.w[0].second);
  }
  return total_size < Arch::LMEM_BYTES;
}

int64_t Cv18xxCycleCalculator::getLocalLayerCycle(Operation *op,
                                                  TensorInfo &tensor_infos,
                                                  group_type_t group_type,
                                                  bool calc_bdc_slack) {
  if (!check_lmem(op, tensor_infos, group_type)) {
    return std::numeric_limits<int64_t>::max() / 100;
  }
  local_sec_info_t sec_info;
  set_local_sec_info(sec_info, op, tensor_infos, group_type);
  std::vector<uint8_t> cmdbuf;
  auto lgOp = dyn_cast<LocalGenInterface>(op);
  lgOp.codegen_local_cv18xx(0, 0, 0, 0, group_type, sec_info, 0);
  CV18xx::submit();
  CV18xx::read_cmdbuf(cmdbuf);
  uint64_t cycle = CV18xxProfiling::get_cycle(cmdbuf);
  return cycle;
}

int64_t Cv18xxCycleCalculator::getGdmaCycle(Value v,
                                            tensor_info_t &tensor_info,
                                            group_type_t group_type, Operation* owner_op, int mode) {
  int64_t cycle = 0;
  if (tensor_info.mode == TIMESTEP_LOAD) {
    cycle = getLoadCycle(v, tensor_info, group_type);
  } else {
    cycle = getStoreCycle(v, tensor_info, group_type);
  }
  return cycle;
}

int64_t Cv18xxCycleCalculator::getLoadCycle(Value v,
                                            tensor_info_t &tensor_info,
                                            group_type_t group_type, Operation* owner_op) {
  int64_t n_slice, c_slice, h_slice, d_slice, w_slice;
  auto &si = tensor_info.slice_info;
  get_max_slice_nchdw(si, n_slice, c_slice, h_slice, d_slice, w_slice);
  bool need_bcast = tensor_info.need_bcast;
  bool eu_align = tensor_info.eu_align;
  bool bcompressed = false;
  auto ifmt = CV18xx::getDataType(v);
  auto ofmt = ifmt;
  int64_t N, C, H, W;
  module::getNCHW(v, N, C, H, W);

  auto g_addr = module::getAddress(v);
  auto l_addr = 0;

  bool isNeuron = true;
  if (module::isWeight(module::getOriValue(v))) {
    isNeuron = false;
  }
  if (isNeuron) {
    if (ifmt == CVK_FMT_U8) {
      ifmt = CVK_FMT_I8;
    }
    if (ofmt == CVK_FMT_U8) {
      ofmt = CVK_FMT_I8;
    }
    assert((ifmt == CVK_FMT_BF16 || ifmt == CVK_FMT_I8) &&
           (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8) &&
           "current load neuron only support int8/bf16");
  } else {
    assert(
        (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8 || ofmt == CVK_FMT_U16) &&
        "current load weight only support int8/uint16/bf16");
    if (ofmt == CVK_FMT_U16) {
      ofmt = CVK_FMT_BF16;
    }
    ifmt = ofmt;
  }
  if (need_bcast) {
    cvi_backend_tl_load_stride_broadcast(0, g_addr, l_addr, n_slice, C, h_slice,
                                         w_slice, C, H, W, eu_align, isNeuron,
                                         ifmt, ofmt, bcompressed);
  } else {
    cvi_backend_tl_load_stride(0, g_addr, l_addr, n_slice, C, h_slice, w_slice,
                               C, H, W, false, eu_align, isNeuron, ifmt, ofmt,
                               bcompressed);
  }
  CV18xx::submit();
  std::vector<uint8_t> cmdbuf;
  CV18xx::read_cmdbuf(cmdbuf);
  uint64_t cycle = CV18xxProfiling::get_cycle(cmdbuf);
  return cycle;
}

int64_t Cv18xxCycleCalculator::getStoreCycle(Value v,
                                             const tensor_info_t &tensor_info,
                                             group_type_t group_type) {
  int64_t n_slice, c_slice, h_slice, d_slice, w_slice;
  auto &si = tensor_info.slice_info;
  get_max_slice_nchdw(si, n_slice, c_slice, h_slice, d_slice, w_slice);
  bool eu_align = tensor_info.eu_align;
  auto ifmt = CV18xx::getDataType(v);
  auto ofmt = ifmt;
  int64_t N, C, H, W;
  module::getNCHW(v, N, C, H, W);

  auto g_addr = module::getAddress(v);
  auto l_addr = 0;

  bool isNeuron = true;
  if (module::isWeight(module::getOriValue(v))) {
    isNeuron = false;
  }
  if (isNeuron) {
    if (ifmt == CVK_FMT_U8) {
      ifmt = CVK_FMT_I8;
    }
    if (ofmt == CVK_FMT_U8) {
      ofmt = CVK_FMT_I8;
    }
    assert((ifmt == CVK_FMT_BF16 || ifmt == CVK_FMT_I8) &&
           (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8) &&
           "current load neuron only support int8/bf16");
  } else {
    assert(
        (ofmt == CVK_FMT_BF16 || ofmt == CVK_FMT_I8 || ofmt == CVK_FMT_U16) &&
        "current load weight only support int8/uint16/bf16");
    if (ofmt == CVK_FMT_U16) {
      ofmt = CVK_FMT_BF16;
    }
    ifmt = ofmt;
  }
  cvi_backend_tl_store_stride(0, g_addr, l_addr, n_slice, C, h_slice, w_slice,
                              C, H, W, false, eu_align, isNeuron, ifmt, ofmt);
  CV18xx::submit();
  std::vector<uint8_t> cmdbuf;
  CV18xx::read_cmdbuf(cmdbuf);
  uint64_t cycle = CV18xxProfiling::get_cycle(cmdbuf);
  return cycle;
}

} // namespace tpu
} // namespace tpu_mlir
