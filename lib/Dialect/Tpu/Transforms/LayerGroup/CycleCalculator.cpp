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
#include "tpu_mlir/Interfaces/InplaceInterface.cpp.inc"
#include "tpu_mlir/Interfaces/InplaceInterface.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

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

SmallVector<Operation *> createTempCoreParallelOp(Operation *_op,
                                                  int num_core) {
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

group_cycle_info_t Bm168xCycleCalculator::getGdmaGroupInfo(
    Value v, tensor_info_t &tensor_info, group_type_t group_type,
    int64_t n_step, int64_t c_step, int64_t d_step, int64_t h_step,
    int64_t w_step, int64_t l_addr) {
  group_cycle_info_t ginfo = {0};
  ginfo.out_addr = l_addr;
  auto si = tensor_info.slice_info;
  auto n_slice_v = si.n;
  auto c_slice_v = si.c;
  auto h_slice_v = si.h;
  auto d_slice_v = si.d;
  auto w_slice_v = si.w;
  if (n_slice_v.empty() && c_slice_v.empty() && h_slice_v.empty() &&
      d_slice_v.empty() && w_slice_v.empty()) {
    int64_t n, c, d, h, w;
    ginfo.overstepped = !(n_step == 0 && c_step == 0 && d_step == 0 &&
                          h_step == 0 && w_step == 0);
    module::getNCDHW(v, n, c, d, h, w, group_type);
    ginfo.n_idx = 0;
    ginfo.c_idx = 0;
    ginfo.d_idx = 0;
    ginfo.h_idx = 0;
    ginfo.w_idx = 0;
    ginfo.n_slice = n;
    ginfo.c_slice = c;
    ginfo.d_slice = d;
    ginfo.h_slice = h;
    ginfo.w_slice = w;
  } else {
    if (n_step >= (int64_t)n_slice_v.size() ||
        c_step >= (int64_t)c_slice_v.size() ||
        h_step >= (int64_t)h_slice_v.size() ||
        d_step >= (int64_t)d_slice_v.size() ||
        w_step >= (int64_t)w_slice_v.size()) {
      ginfo.overstepped = true;
      ginfo.n_idx = n_slice_v[n_step % n_slice_v.size()].first;
      ginfo.c_idx = c_slice_v[c_step % c_slice_v.size()].first;
      ginfo.d_idx = d_slice_v[d_step % d_slice_v.size()].first;
      ginfo.h_idx = h_slice_v[h_step % h_slice_v.size()].first;
      ginfo.w_idx = w_slice_v[w_step % w_slice_v.size()].first;
      ginfo.n_slice = n_slice_v[n_step % n_slice_v.size()].second;
      ginfo.c_slice = c_slice_v[c_step % c_slice_v.size()].second;
      ginfo.d_slice = d_slice_v[d_step % d_slice_v.size()].second;
      ginfo.h_slice = h_slice_v[h_step % h_slice_v.size()].second;
      ginfo.w_slice = w_slice_v[w_step % w_slice_v.size()].second;
    } else {
      ginfo.overstepped = false;
      ginfo.n_idx = n_slice_v[n_step].first;
      ginfo.c_idx = c_slice_v[c_step].first;
      ginfo.h_idx = h_slice_v[h_step].first;
      ginfo.d_idx = d_slice_v[d_step].first;
      ginfo.w_idx = w_slice_v[w_step].first;
      ginfo.n_slice = n_slice_v[n_step].second;
      ginfo.c_slice = c_slice_v[c_step].second;
      ginfo.d_slice = d_slice_v[d_step].second;
      ginfo.h_slice = h_slice_v[h_step].second;
      ginfo.w_slice = w_slice_v[w_step].second;
    }
  }
  return ginfo;
}

int64_t Bm168xCycleCalculator::getLoadCycleOpt(Value v,
                                               tensor_info_t &tensor_info,
                                               group_type_t group_type,
                                               group_cycle_info_t &ginfo) {
  // need_info:
  // - n_slice, h_slice, eu_align, g_addr, l_addr
  // - need_bcast, use_3ic
  // TODO: CONCAT
  auto bm168x = BM168x::instance();
  auto n_idx = ginfo.n_idx;
  auto c_idx = ginfo.c_idx;
  auto d_idx = ginfo.d_idx;
  auto h_idx = ginfo.h_idx;
  auto w_idx = ginfo.w_idx;
  auto n_slice = ginfo.n_slice;
  auto c_slice = ginfo.c_slice;
  auto d_slice = ginfo.d_slice;
  auto h_slice = ginfo.h_slice;
  auto w_slice = ginfo.w_slice;
  auto l_addr = ginfo.out_addr;
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
  if (use_3ic < 4 && use_3ic > 0) {
    // correspoding to NEURON_3IC
    auto g_stride = bm168x->getGlobalStride(N, C, H, W);
    if (need_bcast) {
      c_slice = Arch::NPU_NUM;
      g_stride.N = 0;
      g_stride.C = 0;
      g_stride.H = 0;
    }
    auto l_stride = bm168x->getLocalStride(n_slice, c_slice, h_slice, w_slice,
                                           fmt_bytes, eu_align);
    int64_t g_offset = (n_idx * g_stride.N + c_idx * g_stride.C +
                        h_idx * g_stride.H + w_idx * g_stride.W) *
                       fmt_bytes;
    auto use_op = *v.getUsers().begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    auto kernel = module::getI64Array(conv_op.getKernelShape());
    int64_t to_ic =
        use_3ic == 1
            ? kernel->at(0)
            : (use_3ic == 2 ? kernel->at(1) : kernel->at(0) * kernel->at(1));
    assert(c_slice * to_ic <= Arch::NPU_NUM);
    for (int64_t i = 0; i < c_slice; ++i) {
      bm168x->dl_tensor_broadcast_move_gen_cmd(
          g_addr + g_offset + i * W * H * fmt_bytes, 0, l_addr, i * to_ic,
          n_slice, h_slice, w_slice, to_ic, g_stride.N, g_stride.H, l_stride.N,
          l_stride.H, gdma_format, true, GDMA_VALUE_DIR_S2L, pid_node);
    }
  } else {
    // correspoding to NEURON
    int64_t c_num_local = ceiling_func(c_slice, Arch::NPU_NUM);
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
          int64_t real_c_num_local =
              (channel_index * (int64_t)MAX_TPU_DIM) / Arch::NPU_NUM;
          int64_t dst_offset_c = real_c_num_local * c_stride * fmt_bytes;
          int64_t real_npu_idx =
              (channel_index * (int64_t)MAX_TPU_DIM) % Arch::NPU_NUM;
          int64_t cur_local_offset =
              d * n_slice * c_num_local * c_stride * fmt_bytes + dst_offset_c;
          int64_t src_offset_c =
              (channel_index * (int64_t)MAX_TPU_DIM + c_idx) * H * W *
              fmt_bytes;
          int64_t cur_global_offset = n_idx * C * D * H * W * fmt_bytes +
                                      (d_idx + d) * H * W * fmt_bytes +
                                      h_idx * W * fmt_bytes +
                                      w_idx * fmt_bytes + src_offset_c;
          bm168x->dl_tensor_stride_move_gen_cmd(
              l_addr + cur_local_offset, real_npu_idx,
              g_addr + cur_global_offset, n_slice, cur_cslice, h_slice, w_slice,
              C * D * H * W, D * H * W, W, 1, c_num_local * c_stride, c_stride,
              w_slice, 1, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
          channel_index++;
        }
      }      // depth loop
    } else { // HAVE DEPTH,3D [N,C,D,H,W]->[d,n_slice,c,h_slice,w]
      for (int64_t i = 0; i < n_slice; i++) {
        int64_t cur_local_offset = i * c_num_local * c_stride * fmt_bytes;
        int64_t cur_global_offset = (n_idx + i) * C * D * H * W * fmt_bytes +
                                    c_idx * D * H * W * fmt_bytes +
                                    d_idx * H * W * fmt_bytes +
                                    h_idx * W * fmt_bytes + w_idx * fmt_bytes;
        bm168x->dl_tensor_stride_move_gen_cmd(
            l_addr + cur_local_offset, 0, g_addr + cur_global_offset, d_slice,
            c_slice, h_slice, w_slice,
            H * W,     // actually global d_stride
            D * H * W, // actually global c_stride
            W, 1,
            n_slice * c_num_local * c_stride, // actually local d_stride
            c_stride, w_slice, 1, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
      } // nslice loop
    }
  }
  int64_t gdma_cycle = bm168x->dl_get_cmd_id_cycle(pid_node);
  bm168x->dl_destroy_cmd_id_node(pid_node);
  // llvm::dbgs() << "Load Cycle: " << gdma_cycle << "\n";
  return gdma_cycle;
}

int64_t Bm168xCycleCalculator::getStoreCycleOpt(Value v,
                                                tensor_info_t &tensor_info,
                                                group_type_t group_type,
                                                group_cycle_info_t &ginfo) {
  // need_info:
  // - n_slice, h_slice, eu_align, g_addr, l_addr
  // TODO: CONCAT BMNET_REORG
  auto bm168x = BM168x::instance();
  auto n_idx = ginfo.n_idx;
  auto c_idx = ginfo.c_idx;
  auto d_idx = ginfo.d_idx;
  auto h_idx = ginfo.h_idx;
  auto w_idx = ginfo.w_idx;
  auto n_slice = ginfo.n_slice;
  auto c_slice = ginfo.c_slice;
  auto d_slice = ginfo.d_slice;
  auto h_slice = ginfo.h_slice;
  auto w_slice = ginfo.w_slice;
  auto l_addr = ginfo.out_addr;
  auto pid_node = (CMD_ID_NODE *)bm168x->dl_create_cmd_id_node();
  bm168x->dl_reset_cmd_id(pid_node);
  auto data_type = BM168x::getDataType(v);
  auto gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  bool eu_align = tensor_info.eu_align;
  int64_t N, C, D, H, W;
  module::getNCDHW(v, N, C, D, H, W, group_type);
  auto g_addr = module::getAddress(v);

  int64_t c_num_local = ceiling_func(c_slice, Arch::NPU_NUM);
  int64_t c_stride = eu_align
                         ? align_up(h_slice * w_slice, Arch::eu_num(fmt_bytes))
                         : h_slice * w_slice;
  int64_t channel_num = c_slice;

  if (d_slice <= n_slice) {
    const int64_t csecs = ceiling_func(channel_num, (int64_t)MAX_TPU_DIM);
    for (int64_t d = 0; d < d_slice; d++) {
      int64_t channel_index = 0;
      while (channel_index < csecs) {
        int64_t cur_cslice =
            std::min(channel_num - channel_index * (int64_t)MAX_TPU_DIM,
                     (int64_t)MAX_TPU_DIM);
        int64_t real_c_num_local =
            (channel_index * (int64_t)MAX_TPU_DIM) / Arch::NPU_NUM;
        int64_t src_offset_c = real_c_num_local * c_stride * fmt_bytes;
        int64_t real_npu_idx =
            (channel_index * (int64_t)MAX_TPU_DIM) % Arch::NPU_NUM;
        int64_t cur_local_offset =
            d * n_slice * c_num_local * c_stride * fmt_bytes + src_offset_c;
        int64_t dst_offset_c =
            (channel_index * (int64_t)MAX_TPU_DIM + c_idx) * H * W * fmt_bytes;
        int64_t cur_global_offset = n_idx * C * D * H * W * fmt_bytes +
                                    (d_idx + d) * H * W * fmt_bytes +
                                    h_idx * W * fmt_bytes + w_idx * fmt_bytes +
                                    dst_offset_c;
        bm168x->dl_tensor_stride_move_gen_cmd(
            l_addr + cur_local_offset, real_npu_idx, g_addr + cur_global_offset,
            n_slice, cur_cslice, h_slice, w_slice, c_num_local * c_stride,
            c_stride, w_slice, 1, C * D * H * W, D * H * W, W, 1, gdma_format,
            GDMA_VALUE_DIR_L2S, // 1,
            0, pid_node);
        channel_index++;
      }
    }
  } else { // HAVE DEPTH,3D [D,n_slice,C,h_slice,W] -> [N,C,D,H,W]
    for (int64_t i = 0; i < n_slice; i++) {
      int64_t cur_local_offset = i * c_num_local * c_stride * fmt_bytes;
      int64_t cur_global_offset = (n_idx + i) * C * D * H * W * fmt_bytes +
                                  c_idx * D * H * W * fmt_bytes +
                                  d_idx * H * W * fmt_bytes +
                                  h_idx * W * fmt_bytes + w_idx * fmt_bytes;
      bm168x->dl_tensor_stride_move_gen_cmd(
          l_addr + cur_local_offset, 0, g_addr + cur_global_offset, d_slice,
          c_slice, h_slice, w_slice, n_slice * c_num_local * c_stride, c_stride,
          w_slice, 1, H * W, D * H * W, W, 1, gdma_format,
          GDMA_VALUE_DIR_L2S, // 1,
          0, pid_node);
    }
  }

  int64_t gdma_cycle = bm168x->dl_get_cmd_id_cycle(pid_node);
  bm168x->dl_destroy_cmd_id_node(pid_node);
  // llvm::dbgs() << "Store Cycle: " << gdma_cycle << "\n";
  return gdma_cycle;
}

int64_t Bm168xCycleCalculator::getLocalLayerCycleOpt(
    BasicTimeStepPtr &time_step, Operation *op, TensorInfo &tensor_infos,
    group_type_t group_type, bool calc_bdc_slack, int64_t n_step,
    int64_t c_step, int64_t d_step, int64_t h_step, int64_t w_step) {
  // llvm::dbgs() << "getLocalLayerCycleOpt: "
  //              << "; n_step = " << n_step << "; c_step = " << c_step
  //              << "; d_step = " << d_step << "; h_step = " << h_step
  //              << "; w_step = " << w_step << "; group_type = " << group_type
  //              << "\n";
  auto bm168x = BM168x::instance();
  int64_t cycle = 0;
  local_sec_info_t sec_info;
  auto lgOp = dyn_cast<LocalGenInterface>(op);
  lgOp.lg_assign_sec_info(n_step, c_step, h_step, d_step, w_step, group_type,
                          sec_info, time_step);
  // #pragma omp critical
  {
    bm168x->set_command_issue_flag(false);
    bm168x->reset_cmd_id_node();
    DEBUG_WITH_TYPE("cycle_calc_cmd", {
      llvm::dbgs() << "; action = codegen_local_layer"
                   << "; op_name = " << module::getName(op)
                   << "; tiu_dma_id(before) = "
                   << ((int *)((*BM168x::instance())->bdc_node))[1] << "\n";
    });
    // llvm::dbgs() << "; action = assign_sec_info"
    //              << "; op_name = " << module::getName(op)
    //              << "; n_step = " << n_step << "; c_step = " << c_step
    //              << "; d_step = " << d_step << "; h_step = " << h_step
    //              << "; w_step = " << w_step << "; group_type = " <<
    //              group_type
    //              << "; sec_info.n_slice = " << sec_info.n_slice
    //              << "; sec_info.c_slice = " << sec_info.c_slice
    //              << "; sec_info.d_slice = " << sec_info.d_slice
    //              << "; sec_info.h_slice = " << sec_info.h_slice
    //              << "; sec_info.w_slice = " << sec_info.w_slice
    //              << "; sec_info.n_idx = " << sec_info.n_idx
    //              << "; sec_info.c_idx = " << sec_info.c_idx
    //              << "; sec_info.d_idx = " << sec_info.d_idx
    //              << "; sec_info.h_idx = " << sec_info.h_idx
    //              << "; sec_info.w_idx = " << sec_info.w_idx << "\n";
    lgOp.lg_codegen_local_bm1684x(n_step, c_step, h_step, d_step, w_step,
                                  group_type, sec_info, time_step);

    DEBUG_WITH_TYPE("cycle_calc_cmd", {
      llvm::dbgs() << "; action = codegen_local_layer"
                   << "; op_name = " << module::getName(op)
                   << "; tiu_dma_id(after) = "
                   << bm168x->get_total_id("tiu:0:0")
                   << "; tiu_dma_id(before) = "
                   << ((int *)((*BM168x::instance())->bdc_node))[1] << "\n";
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
  // llvm::dbgs() << "Layer Cycle: " << cycle << "; Layer type: " <<
  // op->getName()
  //              << "\n";
  return cycle;
}

int64_t Bm168xCycleCalculator::getGdmaCycleOpt(Value v,
                                               tensor_info_t &tensor_info,
                                               group_type_t group_type,
                                               group_cycle_info_t &ginfo) {
  auto bm168x = BM168x::instance();
  bm168x->set_command_issue_flag(false);
  bm168x->reset_cmd_id_node();

  // because LoadOp/StoreOp are not created during LayerGroup
  int64_t cycle = 0;
  if (tensor_info.mode == TIMESTEP_LOAD) {
    cycle = getLoadCycleOpt(v, tensor_info, group_type, ginfo);
  } else {
    cycle = getStoreCycleOpt(v, tensor_info, group_type, ginfo);
  }
  bm168x->dl_sg_stas_reset();
  return cycle;
}

int64_t Bm168xCycleCalculator::getGroupCycle(BasicTimeStepPtr &time_step,
                                             shape_secs_t &shape_secs,
                                             group_type_t group_type) {
  if (num_core_ != 1 || !module::isBM1684XFamily()) {
    int64_t total_cycle = 0;
    total_cycle =
        CycleCalculator::getGroupCycle(time_step, shape_secs, group_type);
    return total_cycle;
  }
  int64_t stage_idx = 0;
  int64_t draining_idx = 0;
  bool draining_period = false;
  SoftwarePipeline timestep_swpipl;
  int64_t swpipl_stage_num = time_step->get_swpipl_stage_num();
  int64_t timestep_num = time_step->get_timestep_num();
  auto &tensor_infos = time_step->get_tensor_infos();
  auto nsecs = shape_secs.nsecs;
  auto csecs = shape_secs.csecs;
  auto dsecs = shape_secs.dsecs;
  auto hsecs = shape_secs.hsecs;
  auto wsecs = shape_secs.wsecs;
  // int64_t base_cycle = 299647;
  int64_t layer_cycle = 0;
  int64_t gdma_cycle = 0;
  int64_t total_cycle = 0;
  for (uint64_t nstep = 0, cstep = 0, dstep = 0, hstep = 0, wstep = 0;
       nstep < nsecs || draining_period;) {
    // llvm::dbgs() << "Stage: " << stage_idx
    //              << "; draining_period: " << draining_period
    //              << "; draining_idx: " << draining_idx
    //              << "; nstep: " << nstep << "; cstep: " << cstep
    //              << "; dstep: " << dstep << "; hstep: " << hstep
    //              << "; wstep: " << wstep << "\n";
    /* add for software pipeline */
    timestep_swpipl.write_swloop_buffer(nstep, cstep, hstep, dstep, wstep,
                                        swpipl_stage_num);
    for (int64_t ts = 0; ts < timestep_num; ++ts) {
      const TpuTsField &timestep_layers = time_step->getLayers(ts);
      const GdmaTsField &timestep_tensors = time_step->getTensors(ts);

      for (auto tensor : timestep_tensors) {
        auto op_stage = time_step->get_tensor_swpipl_stage(tensor.first);
        if ((!draining_period && op_stage > stage_idx) ||
            (draining_period &&
             (op_stage < draining_idx || op_stage > stage_idx))) {
          // llvm::dbgs() << "Skipping tensor: " << tensor.first
          //            << " at stage: " << op_stage
          //             << " during draining period: " << draining_period
          //             << " with draining_idx: " << draining_idx
          //             << " and stage_idx: " << stage_idx << "\n";
          continue;
        }
        const tensor_step_t *tensor_step =
            timestep_swpipl.read_swloop_buffer(op_stage);
        auto tensor_info = tensor.second;
        // if (module::isBM1688()) {
        //   auto bm1688 = (BM1688 *)BM168x::instance();
        //   float BW = 24;
        //   if (consider_multi_core_bw) {
        //     BW = 15.f;
        //     DEBUG_WITH_TYPE("cycle_calc", {
        //       llvm::dbgs() << "; action = multi_core_align"
        //                   << "; BW = " << BW << "\n";
        //     });
        //     bm1688->dl_set_gdma_bw_s2l(BW);
        //     bm1688->dl_set_gdma_bw_l2s(BW);
        //   }
        // }
        mem_buffer_key_t buffer_key;
        buffer_key.value = tensor.first;
        buffer_key.type =
            module::isWeight(tensor.first) ? LMEM_WEIGHT : LMEM_ACTIVATION;
        auto l_addr = time_step->get_lmem_addr(buffer_key);
        auto ginfo = this->getGdmaGroupInfo(
            tensor.first, tensor_info, group_type, tensor_step->nstep,
            tensor_step->cstep, tensor_step->dstep, tensor_step->hstep,
            tensor_step->wstep, l_addr);
        if (ginfo.overstepped == false || stage_idx == op_stage) {
          ginfo.overstepped = true;
          // int64_t cycle =
          //     this->getGdmaCycleOpt(tensor.first, tensor_info, group_type,
          //     n_step,
          //                           c_step, d_step, h_step, w_step, l_addr);
          int64_t cycle = this->getGdmaCycleOpt(tensor.first, tensor_info,
                                                group_type, ginfo);
          gdma_cycle += cycle;
          // llvm::dbgs() << "gdma cycle_count: " << gdma_cycle + base_cycle <<
          // "\n";
        }
      }
      for (auto op : timestep_layers) {
        auto op_stage = time_step->get_layer_swpipl_stage(op);
        if ((!draining_period && op_stage > stage_idx) ||
            (draining_period &&
             (op_stage < draining_idx || op_stage > stage_idx))) {
          continue;
        }
        const tensor_step_t *tensor_step =
            timestep_swpipl.read_swloop_buffer(op_stage);
        int64_t cycle = this->getLocalLayerCycleOpt(
            time_step, op, tensor_infos, group_type, false, tensor_step->nstep,
            tensor_step->cstep, tensor_step->dstep, tensor_step->hstep,
            tensor_step->wstep);
        layer_cycle += cycle;
        // llvm::dbgs() << "layer cycle_count: " << layer_cycle + base_cycle <<
        // "\n";
      }
      total_cycle = std::max(layer_cycle, gdma_cycle);
      // llvm::dbgs() << "total cycle_count: " << total_cycle + base_cycle <<
      // "\n";
      layer_cycle = total_cycle;
      gdma_cycle = total_cycle;
    }
    if (!draining_period) {
      cstep++;
      if (cstep >= csecs) {
        cstep = 0;
        wstep++;
      }
      if (wstep >= wsecs) {
        wstep = 0;
        hstep++;
      }
      if (hstep >= hsecs) {
        hstep = 0;
        dstep++;
      }
      if (dstep >= dsecs) {
        dstep = 0;
        nstep++;
        if (nstep >= nsecs) { // && swpipl_stage_num > 1
          draining_period = true;
        }
      }
      // if (useMuliCore && ((stage_idx + 1) % max_task_per_core) == 0 &&
      //     nstep < nsecs) {
      //   draining_period = true;
      //   draining_idx = 0;
      // }
    }
    stage_idx++;
    if (draining_period) {
      draining_idx++;
      if (draining_idx >= swpipl_stage_num) {
        draining_period = false;
        stage_idx = 0;
        draining_idx = 0;
      }
    }
  }
  return total_cycle;
}

int64_t CycleCalculator::getGroupCycle(BasicTimeStepPtr &time_step,
                                       shape_secs_t &shape_secs,
                                       group_type_t group_type) {
  // (void*)invokeInIterationSpace;

  int64_t loop_num = shape_secs.nsecs * shape_secs.csecs * shape_secs.hsecs *
                     shape_secs.dsecs * shape_secs.wsecs;

  DEBUG_WITH_TYPE("cycle_calc", {
    llvm::dbgs() << "; action = cycle_calc"
                 << "; loop_num = " << loop_num << "\n";
  });
  std::vector<layer_cycle_info_t> layer_cycle;
  std::vector<gdma_cycle_info_t> gdma_cycle;
  bool consider_multi_core_bw = false;
  if (num_core_ == 2) {
    consider_multi_core_bw = true;
    loop_num = loop_num / num_core_ + (loop_num % num_core_ > 0);
    DEBUG_WITH_TYPE("cycle_calc", {
      llvm::dbgs() << "; action = cycle_calc"
                   << "; step = multi_core_refactor"
                   << "; loop_num = " << loop_num << "\n";
    });
  }

  if (num_core_ == 8) {
    loop_num = loop_num / num_core_ + (loop_num % num_core_ > 0);
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
      if (module::isBM1688()) {
        auto bm1688 = (BM1688 *)BM168x::instance();
        float BW = 24;
        if (consider_multi_core_bw) {
          BW = 15.f;
          DEBUG_WITH_TYPE("cycle_calc", {
            llvm::dbgs() << "; action = multi_core_align"
                         << "; BW = " << BW << "\n";
          });
          bm1688->dl_set_gdma_bw_s2l(BW);
          bm1688->dl_set_gdma_bw_l2s(BW);
        }
      }
      int64_t cycle = this->getGdmaCycle(tensor.first, tensor_info, group_type);
      if (module::isBM1688()) {
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
                     << "; hold_in_lmem = " << hold_in_lmem
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
                         << "; value = " << layer.cycle << "\n";
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
                         << "; value = " << tensor.cycle << "\n";
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
                       << "; value = " << total_layer_cycle << "\n";
        } else {
          llvm::dbgs() << "; action = consider_gdma_cycle"
                       << "; stage = filling"
                       << "; value = " << total_gdma_cycle << "\n";
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
                       << "; value = " << layer.cycle << "\n";
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
                         << "; value = " << tensor.cycle << "\n";
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
                         << "; value = " << layer.cycle << "\n";
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
                         << "; value = " << tensor.cycle << "\n";
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
                 << "; event = "
                 << "total_cycle_of_group"
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
  auto splitedOps = createTempCoreParallelOp(op, num_core_);
  // SmallVector<Operation *> splitedOps;

  if (auto inplaceOp = dyn_cast<InplaceInterface>(op)) {
    if (inplaceOp.SupportInplace()) {
      return 0;
    }
  }

  if (splitedOps.size() == 0) {
    splitedOps.push_back(op);
  }

  auto bm168x = BM168x::instance();

  if (module::isBM1688()) {
    auto bm1688 = (BM1688 *)bm168x;
    bool imp_multi_core_global =
        isa<tpu::Conv2DOp>(op) || splitedOps.size() > 1;
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
      auto full_cycle = bm168x->get_cmd_cycle();
      auto op_name = module::getName(_op);
      DEBUG_WITH_TYPE("cycle_calc", {
        llvm::dbgs() << "; action = codegen_global_layer"
                     << "; op_name = " << op_name << "; full = " << full_cycle
                     << "\n";
      });
      cycle = std::max(cycle, full_cycle);
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
                     << "; core_parallel = " << cycle << "\n";
      });
      cycle = std::min(cycle, full_cycle);
      bm168x->dl_sg_stas_reset();
    }
  });

  if (splitedOps.size() > 1) {
    removeTempCoreParallelOp(splitedOps);
  }

  // restore default BW
  if (module::isBM1688()) {
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
                   << "; tiu_dma_id(before) = "
                   << ((int *)((*BM168x::instance())->bdc_node))[1] << "\n";
    });
    // set_local_layer_io_addr(op);
    lgOp.codegen_local_bm168x(0, 0, 0, 0, 0, group_type, sec_info);

    DEBUG_WITH_TYPE("cycle_calc_cmd", {
      llvm::dbgs() << "; action = codegen_local_layer"
                   << "; op_name = " << module::getName(op)
                   << "; tiu_dma_id(after) = "
                   << bm168x->get_total_id("tiu:0:0")
                   << "; tiu_dma_id(before) = "
                   << ((int *)((*BM168x::instance())->bdc_node))[1] << "\n";
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

int64_t Bm168xCycleCalculator::getGdmaCycle(Value v, tensor_info_t &tensor_info,
                                            group_type_t group_type,
                                            Operation *owner_op, int mode) {
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

int64_t Bm168xCycleCalculator::getLoadCycle(Value v, tensor_info_t &tensor_info,
                                            group_type_t group_type,
                                            Operation *owner_op) {
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
  std::vector<slice_pair_t> slice_idx = get_max_slice_nchdw_and_idx(
      si, n_slice, c_slice, h_slice, d_slice, w_slice);
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
              (uint64_t)(slice_idx[2].second) * W * data_len +
              slice_idx[4].second * data_len,
          0, // no use
          src_N, src_C, src_H, src_W, src_N_stride, src_C_stride, src_H_stride,
          src_W_stride, 0, local_offerset + i * dst_N_stride * data_len, 0,
          dst_N, dst_C, dst_H, dst_W, dst_N_stride, dst_C_stride, dst_H_stride,
          dst_W_stride, 0,
          GDMA_VALUE_DIR_S2L, // 0
          0, pid_node);
    }
  } else {
    if (use_3ic < 4 && use_3ic > 0) {
      // correspoding to NEURON_3IC
      auto g_stride = bm168x->getGlobalStride(N, C, H, W);
      if (need_bcast) {
        c_slice = Arch::NPU_NUM;
        g_stride.N = 0;
        g_stride.C = 0;
        g_stride.H = 0;
      }
      auto l_stride = bm168x->getLocalStride(n_slice, c_slice, h_slice, w_slice,
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
      int64_t c_num_local = ceiling_func(c_slice, Arch::NPU_NUM);
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
      if (si.h[0].second > (4095 - 32)) {
        return false;
      }
      total_size += Arch::get_tensor_lmem_bytes(in, si.n[0].second,
                                                si.c[0].second, si.d[0].second,
                                                si.h[0].second, si.w[0].second);
    }
  }
  for (auto out : outs) {
    auto iter = tensor_infos.find(out);
    if (iter == tensor_infos.end())
      continue;
    auto &si = iter->second.slice_info;
    if (si.h[0].second > (4095 - 32)) {
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

int64_t Cv18xxCycleCalculator::getGdmaCycle(Value v, tensor_info_t &tensor_info,
                                            group_type_t group_type,
                                            Operation *owner_op, int mode) {
  int64_t cycle = 0;
  if (tensor_info.mode == TIMESTEP_LOAD) {
    cycle = getLoadCycle(v, tensor_info, group_type);
  } else {
    cycle = getStoreCycle(v, tensor_info, group_type);
  }
  return cycle;
}

int64_t Cv18xxCycleCalculator::getLoadCycle(Value v, tensor_info_t &tensor_info,
                                            group_type_t group_type,
                                            Operation *owner_op) {
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
