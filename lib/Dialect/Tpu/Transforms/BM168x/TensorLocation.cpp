//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/TensorLocation.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <llvm/Support/JSON.h>
#include <variant>

namespace mlir {
using namespace llvm;
using namespace tpu_mlir::backend;

std::shared_ptr<TensorLocationImpl> TensorLocation::impl;

struct slice {
  int64_t begin;
  int64_t step;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const slice &S) {
  if (S.begin >= 0)
    OS << S.begin;
  OS << ":";
  if (S.step >= 0)
    OS << S.begin + S.step;
  return OS;
}

inline std::string fmt_slice(std::initializer_list<slice> slices) {
  std::string stride;
  llvm::raw_string_ostream os(stride);
  os << "[";
  interleave(slices, os, ", ");
  os << "]";
  return stride;
}

template <typename T>
inline std::string fmt_shape(ArrayRef<T> shape, StringRef dtype = {}) {
  std::string shape_str;
  llvm::raw_string_ostream os(shape_str);
  os << "<";
  interleave(shape, os, "x");
  if (dtype.size() > 0)
    os << "x" << dtype;
  os << ">";
  return shape_str;
}

inline std::string get_dtype_str(int dtype) {
  assert(dtype >= 0);
  std::string dtype_str[] = {"f32",  "f16",  "si8",  "ui8", "si16", "ui16",
                             "si32", "ui32", "bf16", "si4", "ui4"};
  return dtype_str[dtype];
}

struct slice_index {
  int64_t n_step;
  int64_t h_step;
  int64_t d_step;
  int64_t w_step;
};

group_info_t getGroupInfo(Value v, const slice_index &slice_i) {
  return LocalGenInterface::getGroupInfo(v, slice_i.n_step, slice_i.h_step,
                                         slice_i.d_step, slice_i.w_step);
}

group_info_t getGroupInfo(Operation *op, const slice_index &slice_i) {
  return LocalGenInterface::getGroupInfo(op, slice_i.n_step, slice_i.h_step,
                                         slice_i.d_step, slice_i.w_step);
}

group_info_t getGroupInfo(const OpOperand &v, const slice_index &slice_i) {
  if (auto op = v.get().getDefiningOp())
    if (op->hasAttr(LocalGenInterface::kLayerGroupAttrName))
      return getGroupInfo(op, slice_i);

  // Computing the backward slice is time-consuming.
  group_info_t ginfo = {0};
  auto dst_op = v.getOwner();
  auto dst_lg_op = cast<LocalGenInterface>(dst_op);
  auto g_param = dst_op->getAttr(LocalGenInterface::kLayerGroupAttrName)
                     .cast<tpu::LayerGroupAttr>();
  int64_t nslice = g_param.getNSlice()[slice_i.n_step];
  int64_t hslice = g_param.getHSlice()[slice_i.h_step];
  int64_t dslice = g_param.getDSlice()[slice_i.d_step];
  int64_t wslice = g_param.getWSlice()[slice_i.w_step];
  int64_t nindex = g_param.getNIdx()[slice_i.n_step];
  int64_t hindex = g_param.getHIdx()[slice_i.h_step];
  int64_t dindex = g_param.getDIdx()[slice_i.d_step];
  int64_t windex = g_param.getWIdx()[slice_i.w_step];
  dst_lg_op.BackwardN(ginfo.n_idx, ginfo.n_slice, nindex, nslice);
  dst_lg_op.BackwardH(ginfo.h_idx, ginfo.h_slice, hindex, hslice);
  dst_lg_op.BackwardD(ginfo.d_idx, ginfo.d_slice, dindex, dslice);
  dst_lg_op.BackwardW(ginfo.w_idx, ginfo.w_slice, windex, wslice);
  return ginfo;
}

template <typename T>
json::Object record_tensor(const T val_or_opd, const slice_index &slice_i,
                           group_type_t group_type) {

  group_info_t ginfo;
  Value val;
  if constexpr (std::is_same<T, OpOperand *>::value) {
    val = val_or_opd->get();
    ginfo = getGroupInfo(*val_or_opd, slice_i);
  } else {
    static_assert(std::is_same<T, OpResult>::value,
                  "Wrong Type! only supports OpOperand and OpResult.");
    val = val_or_opd;
    ginfo = getGroupInfo(val, slice_i);
  };

  auto v_spc = BM168x::value_to_spec(val, group_type);
  auto name = module::getName(val);

  std::string type;
  llvm::raw_string_ostream os(type);
  val.getType().print(os);

  int64_t re_shape[5];
  module::getNCDHW(val, re_shape[0], re_shape[1], re_shape[2], re_shape[3],
                   re_shape[4], group_type);
  auto reshape = fmt_shape(ArrayRef(re_shape));

  int64_t mem_shape[] = {ginfo.n_slice * ginfo.d_slice, re_shape[1],
                         ginfo.h_slice, ginfo.w_slice};

  auto memory_type = fmt_shape(ArrayRef(mem_shape), get_dtype_str(v_spc.dtype));

  uint64_t address = v_spc.addr;

  auto slice = fmt_slice({{ginfo.n_idx, ginfo.n_slice},
                          {-1, -1},
                          {ginfo.d_idx, ginfo.d_slice},
                          {ginfo.h_idx, ginfo.h_slice},
                          {ginfo.w_idx, ginfo.w_slice}});

  std::string layout = ginfo.eu_align ? "eu_align" : "compact";

  { // global memory Load operand/Store result
    auto offset = [&]() -> int64_t {
      auto fmt_bytes = BM168x::getFmtBytes((DATA_TYPE_T)v_spc.dtype);
      SmallVector<int64_t> stride;
      stride.push_back(1);
      for (auto i : llvm::reverse(re_shape)) {
        stride.push_back(i * stride.back());
      }
      int64_t idx[] = {ginfo.w_idx, ginfo.h_idx, ginfo.d_idx, 0, ginfo.n_idx};
      int64_t offset = 0;
      for (int i = 0; i < 5; i++) {
        offset += stride[i] * idx[i];
      }
      return offset * fmt_bytes;
    };

    auto op = val.getDefiningOp();
    if (op == nullptr || !op->hasAttr(LocalGenInterface::kLayerGroupAttrName)) {
      address += offset();
      layout = "continuous";
    } else if (isa_and_nonnull<tpu::StoreOp>(op)) {
      address = module::getAddress(val) + offset();
      layout = "continuous";
    }
  }

  if (group_type == GROUP_3D) {
    layout += "_group3d"; // {d * n, c, h, w}
  }

  return json::Object{
      {"name", name},     {"address", address}, {"memory_type", memory_type},
      {"layout", layout}, {"type", type},       {"reshape", reshape},
      {"slice", slice},
  };
}

json::Object record_tensor(Value v, const group_type_t group_type) {
  auto v_spc = BM168x::value_to_spec(v, group_type);
  std::string type;
  llvm::raw_string_ostream os(type);
  v.getType().print(os);

  auto memory_type =
      fmt_shape(ArrayRef(v_spc.shape, v_spc.dims), get_dtype_str(v_spc.dtype));

  uint64_t address = v_spc.addr;

  return json::Object{{"name", module::getName(v).str()},
                      {"address", address},
                      {"memory_type", memory_type},
                      {"layout", "continuous"},
                      {"type", type},
                      {"reshape", ""},
                      {"slice", "[...]"}};
}

void TensorLocationImpl::record_loc(Operation *op, const json::Array &operands,
                                    const json::Array &results) {
  int64_t line_num = -1; // unknown location
  auto it = opToLineCol.find(op);
  if (it != opToLineCol.end()) {
    line_num = it->second.first;
  }
  int subnet_id = -1;
  if (auto func = dyn_cast<func::FuncOp>(op->getParentOp()))
    subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  else if (auto lg = dyn_cast<tpu::GroupOp>(op->getParentOp())) {
    auto func = cast<func::FuncOp>(lg->getParentOp());
    subnet_id = func->getAttrOfType<IntegerAttr>("id").getInt();
  }

  J.object([&] {
    J.attribute("file-line", line_num);
    J.attribute("subnet_id", subnet_id);
    J.attribute("opcode", op->getName().getStringRef());
    J.attributeArray("bdc_gdma_id(before)", [&] {
      J.value(cmd_before[0]);
      J.value(cmd_before[1]);
    });
    J.attributeArray("bdc_gdma_id(after)", [&] {
      J.value(BM168x::instance()->bdc_total_id);
      J.value(BM168x::instance()->gdma_total_id);
    });
    J.attributeArray("operands", [&] {
      for (auto &v : operands)
        J.value(v);
    });
    J.attributeArray("results", [&] {
      for (auto &v : results)
        J.value(v);
    });
  });
}

void TensorLocationImpl::after_codegen_local(Operation *op, int64_t n_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  auto slice_i = slice_index{
      .n_step = n_step, .h_step = h_step, .d_step = d_step, .w_step = w_step};

  json::Array operands, results;

  for (auto &v : op->getOpOperands()) {
    if (module::isNone(v.get()))
      operands.push_back(json::Object());
    else
      operands.push_back(record_tensor(&v, slice_i, group_type));
  }

  for (auto v : op->getResults()) {
    if (module::isNone(v)) {
      results.push_back(json::Object());
    } else
      results.push_back(record_tensor(v, slice_i, group_type));
  }
  record_loc(op, operands, results);
}

void TensorLocationImpl::after_codegen_global(Operation *op) {
  json::Array operands, results;

  for (auto v : op->getOperands()) {
    if (module::isNone(v)) {
      operands.push_back(json::Object());
    } else
      operands.push_back(record_tensor(v, GROUP_NORMAL));
  }

  for (auto v : op->getResults()) {
    if (module::isNone(v)) {
      results.push_back(json::Object());
    } else
      results.push_back(record_tensor(v, GROUP_NORMAL));
  }
  record_loc(op, operands, results);
}

} // namespace mlir
