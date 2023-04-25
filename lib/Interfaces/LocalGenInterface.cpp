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
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step, int64_t c_step) {
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
    auto shape = module::getShape(dst_op->getResult(0));
    int64_t nslice = g_param.getNSlice()[0];
    int64_t hslice = g_param.getHSlice()[0];
    int64_t dslice = g_param.getDSlice()[0];
    int64_t wslice = g_param.getWSlice()[0];
    int64_t cslice = g_param.getCSlice()[0];
    dst_lg_op.BackwardN(ginfo.n_idx, ginfo.n_slice, 0, nslice);
    dst_lg_op.BackwardH(ginfo.h_idx, ginfo.h_slice, 0, hslice);
    dst_lg_op.BackwardD(ginfo.d_idx, ginfo.d_slice, 0, dslice);
    if (g_param.getGroupType() == GROUP_MM) {
      dst_lg_op.BackwardC(ginfo.c_idx, ginfo.c_slice, 0, cslice);
    }
    if (shape.size() >= 4) {
      dst_lg_op.BackwardW(ginfo.w_idx, ginfo.w_slice, 0, wslice);
    }
    return ginfo;
  }
  return getGroupInfo(op, n_step, h_step, d_step, w_step, c_step);
}

group_info_t LocalGenInterface::getGroupInfo(mlir::Operation *op,
                                             int64_t n_step, int64_t h_step,
                                             int64_t d_step, int64_t w_step,
                                             int64_t c_step) {
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
  auto d_idx_v = g_param.getDIdx();
  auto d_slice_v = g_param.getDSlice();
  auto w_idx_v = g_param.getWIdx();
  auto w_slice_v = g_param.getWSlice();
  auto c_idx_v = g_param.getCIdx();
  auto c_slice_v = g_param.getCSlice();
  if (n_idx_v.empty() && c_idx_v.empty() && d_idx_v.empty() &&
      h_idx_v.empty() && w_idx_v.empty()) {
    int64_t n, c, d, h, w;
    ginfo.overstepped = !(n_step == 0 && c_step == 0 && d_step == 0 &&
                          h_step == 0 && w_step == 0);
    module::getNCDHW(op->getResult(0), n, c, d, h, w, (group_type_t)ginfo.type);
    ginfo.n_slice = n;
    ginfo.c_slice = c;
    ginfo.h_slice = h;
    ginfo.d_slice = d;
    ginfo.w_slice = w;
  } else {
    if (n_step >= (int64_t)n_idx_v.size() ||
        c_step >= (int64_t)c_idx_v.size() ||
        h_step >= (int64_t)h_idx_v.size() ||
        d_step >= (int64_t)d_idx_v.size() ||
        w_step >= (int64_t)w_idx_v.size()) {
      ginfo.overstepped = true;
    } else {
      ginfo.n_idx = n_idx_v[n_step];
      ginfo.c_idx = c_idx_v[c_step];
      ginfo.h_idx = h_idx_v[h_step];
      ginfo.d_idx = d_idx_v[d_step];
      ginfo.w_idx = w_idx_v[w_step];
      ginfo.n_slice = n_slice_v[n_step];
      ginfo.c_slice = c_slice_v[c_step];
      ginfo.h_slice = h_slice_v[h_step];
      ginfo.d_slice = d_slice_v[d_step];
      ginfo.w_slice = w_slice_v[w_step];
      ginfo.overstepped = false;
    }
  }
  return ginfo;
}

static int bcast_type(int s0, int s1) {
  if (s0 == s1)
    return 0;
  else if (s0 > s1)
    return 1;
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
  if (module::isCV18xx()) {
    if (lhs_shape != rhs_shape) {
      return failure();
    }
    return success();
  }
  if (lhs_shape.size() >= 5) {
    const int wdim = 3;
    int bcast = bcast_type(lhs_shape[wdim], rhs_shape[wdim]);
    for (int i = wdim + 1; i < lhs_shape.size(); ++i) {
      if (bcast != bcast_type(lhs_shape[i], rhs_shape[i])) {
        return failure();
      }
    }
  }

  return success();
}

} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/LocalGenInterface.cpp.inc"
