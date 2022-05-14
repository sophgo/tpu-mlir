#include "sophgo/Interfaces/LayerGroupInterface.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"

using namespace mlir;

namespace sophgo {

constexpr llvm::StringRef LayerGroupInterface::kLayerGroupAttrName;
void LayerGroupInterface::fixSlice(int64_t &in_idx, int64_t &in_slice,
                                   int64_t in_length) {
  // avoid leak
  auto end_idx = in_idx + in_slice;
  if (in_idx < 0) {
    in_idx = 0;
  }
  if (end_idx > in_length) {
    end_idx = in_length;
  }
  in_slice = end_idx - in_idx;
}

group_info_t LayerGroupInterface::getGroupInfo(mlir::Value v, int64_t n_step,
                                               int64_t h_step) {
  return getGroupInfo(v.getDefiningOp(), n_step, h_step);
}

group_info_t LayerGroupInterface::getGroupInfo(mlir::Operation *op,
                                               int64_t n_step, int64_t h_step) {
  assert(op != nullptr);
  assert(op->hasAttr(LayerGroupInterface::kLayerGroupAttrName));
  auto g_param = op->getAttr(LayerGroupInterface::kLayerGroupAttrName)
                     .cast<tpu::LayerGroup>();
  group_info_t ginfo = {0};
  ginfo.out_addr = g_param.out_addr().getInt();
  ginfo.out_size = g_param.out_size().getInt();
  ginfo.buffer_addr = g_param.buffer_addr().getInt();
  ginfo.buffer_size = g_param.buffer_size().getInt();
  ginfo.timestep = g_param.timestep().getInt();
  auto n_idx_v = Module::getI64Array(g_param.n_idx());
  auto n_slice_v = Module::getI64Array(g_param.n_slice());
  auto h_idx_v = Module::getI64Array(g_param.h_idx());
  auto h_slice_v = Module::getI64Array(g_param.h_slice());
  ginfo.n_idx = n_idx_v->at(n_step);
  ginfo.n_slice = n_slice_v->at(n_step);
  ginfo.h_idx = h_idx_v->at(h_step);
  ginfo.h_slice = h_slice_v->at(h_step);
  return ginfo;
}

} // namespace sophgo

#include "sophgo/Interfaces/LayerGroupInterface.cpp.inc"
