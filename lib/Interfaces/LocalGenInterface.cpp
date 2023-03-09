//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;

namespace tpu_mlir {
constexpr llvm::StringRef LocalGenInterface::kLayerGroupAttrName;
void LocalGenInterface::fixSlice(int64_t &in_idx, int64_t &in_slice,
                                 int64_t in_length, bool last) {
  // avoid leak
  auto end_idx = in_idx + in_slice;
  if (in_idx < 0) {
    in_idx = 0;
  }
  if (end_idx > in_length || last) {
    end_idx = in_length;
  }
  in_slice = end_idx - in_idx;
}

group_info_t LocalGenInterface::getGroupInfo(mlir::Value v, int64_t n_step,
                                             int64_t h_step) {
  auto op = v.getDefiningOp();
  if (op == nullptr || !op->hasAttr(LocalGenInterface::kLayerGroupAttrName)) {
    // generate ginfo
    group_info_t ginfo = {0};
    if (v.getType().isa<NoneType>()) {
      return ginfo;
    }
    auto dst_op = *v.getUsers().begin();
    auto dst_lg_op = cast<LocalGenInterface>(dst_op);
    auto g_param = dst_op->getAttr(LocalGenInterface::kLayerGroupAttrName)
                       .cast<tpu::LayerGroupAttr>();
    int64_t nslice = g_param.getNSlice()[0];
    int64_t hslice = g_param.getHSlice()[0];
    dst_lg_op.BackwardN(ginfo.n_idx, ginfo.n_slice, 0, nslice);
    dst_lg_op.BackwardH(ginfo.h_idx, ginfo.h_slice, 0, hslice);
    return ginfo;
  }
  return getGroupInfo(op, n_step, h_step);
}

group_info_t LocalGenInterface::getGroupInfo(mlir::Operation *op,
                                             int64_t n_step, int64_t h_step) {
  group_info_t ginfo = {0};
  if (isa<top::NoneOp>(op)) {
    return ginfo;
  }
  assert(op != nullptr);
  assert(op->hasAttr(LocalGenInterface::kLayerGroupAttrName));
  auto g_param = op->getAttr(LocalGenInterface::kLayerGroupAttrName)
                     .cast<tpu::LayerGroupAttr>();
  ginfo.id = g_param.getId();
  ginfo.stage = g_param.getStage();
  ginfo.out_addr = g_param.getOutAddr();
  ginfo.out_size = g_param.getOutSize();
  ginfo.buffer_addr = g_param.getBufferAddr();
  ginfo.buffer_size = g_param.getBufferSize();
  ginfo.eu_align = g_param.getEuAlign();
  ginfo.type = g_param.getGroupType();
  auto n_idx_v = g_param.getNIdx();
  auto n_slice_v = g_param.getNSlice();
  auto h_idx_v = g_param.getHIdx();
  auto h_slice_v = g_param.getHSlice();
  if (n_idx_v.empty() && h_idx_v.empty()) {
    int64_t n, c, h, w;
    ginfo.overstepped = !(n_step == 0 && h_step == 0);
    module::getNCHW(op->getResult(0), n, c, h, w);
    ginfo.n_slice = n;
    ginfo.h_slice = h;
  } else {
    if (n_step >= (int64_t)n_idx_v.size() ||
        h_step >= (int64_t)h_idx_v.size()) {
      ginfo.overstepped = true;
    } else {
      ginfo.n_idx = n_idx_v[n_step];
      ginfo.n_slice = n_slice_v[n_step];
      ginfo.h_idx = h_idx_v[h_step];
      ginfo.h_slice = h_slice_v[h_step];
      ginfo.overstepped = false;
    }
  }
  return ginfo;
}

static int bcast_type(int s0, int s1){
  if(s0==s1) return 0;
  else if(s0>s1) return 1;
  return -1;
}

LogicalResult BroadCastBinaryLocalGenSupport(Operation *op) {
  auto out_shape = module::getShape(op->getResult(0));
  auto lhs_shape = module::getShape(op->getOperand(0));
  auto rhs_shape = module::getShape(op->getOperand(1));
  if (lhs_shape.size() != rhs_shape.size())
    return failure();
  if (module::isWeight(op->getOperand(0)) ||
      module::isWeight(op->getOperand(1)))
    return failure();
  if (lhs_shape.size() >= 5) {
    const int wdim = 3;
    int bcast = bcast_type(lhs_shape[wdim], rhs_shape[wdim]);
    for (int i=wdim+1; i < lhs_shape.size(); ++i) {
      if (bcast != bcast_type(lhs_shape[i], rhs_shape[i])) {
        return failure();
      }
    }
  }

  return success();
}

} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/LocalGenInterface.cpp.inc"
