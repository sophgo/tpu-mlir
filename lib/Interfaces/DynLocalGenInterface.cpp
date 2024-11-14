//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

namespace tpu_mlir {

constexpr llvm::StringRef DynLocalGenInterface::kLayerGroupAttrName;
group_info_t DynLocalGenInterface::DynGetGroupInfo(mlir::Value v,
                                                   int64_t n_step,
                                                   int64_t h_step) {
  return DynGetGroupInfo(v.getDefiningOp(), n_step, h_step);
}

group_info_t DynLocalGenInterface::DynGetGroupInfo(mlir::Operation *op,
                                                   int64_t n_step,
                                                   int64_t h_step) {
  group_info_t ginfo = {0};
  if (isa<top::NoneOp>(op)) {
    return ginfo;
  }
  assert(op != nullptr);
  assert(op->hasAttr(DynLocalGenInterface::kLayerGroupAttrName));
  auto g_param = op->getAttr(DynLocalGenInterface::kLayerGroupAttrName)
                     .cast<tpu::LayerGroupAttr>();
  ginfo.id = g_param.getId();
  ginfo.stage = g_param.getStage();
  ginfo.out_addr = g_param.getOutAddr();
  ginfo.out_size = g_param.getOutSize();
  ginfo.buffer_addr = g_param.getBufferAddr();
  ginfo.buffer_size = g_param.getBufferSize();
  ginfo.eu_align = g_param.getEuAlign();
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

} // namespace tpu_mlir

#include "tpu_mlir/Interfaces/DynLocalGenInterface.cpp.inc"
