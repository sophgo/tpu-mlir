//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/TensorLocation.hpp"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <llvm/Support/JSON.h>

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

inline std::string Slice(std::initializer_list<slice> slices) {
  std::string stride;
  llvm::raw_string_ostream os(stride);
  os << "[";
  interleave(slices, os, ", ");
  os << "]";
  return stride;
}

json::Object record_tensor(Value v, const group_type_t &group_type,
                           const StringRef slice, const StringRef layout) {
  auto v_spc = BM168x::value_to_spec(v, group_type);
  std::string type, shape, S;
  llvm::raw_string_ostream os(S);
  v.getType().print(os);
  type = S;
  S.clear();
  os << "<";
  interleave(llvm::ArrayRef(v_spc.shape, v_spc.dims), os, "x");
  os << ">";
  shape = S;
  uint64_t address = v_spc.addr;
  if (isa_and_nonnull<tpu::StoreOp>(v.getDefiningOp()))
    address = module::getAddress(v);

  return json::Object{{"name", module::getName(v).str()},
                      {"type", type},
                      {"shape", shape},
                      {"slice", slice},
                      {"address", address},
                      {"layout", layout}};
}

void TensorLocationImpl::record_loc(Operation *op, const json::Array &operands,
                                    const json::Array &results) {
  int64_t line_num = -1; // unknown location
  auto it = opToLineCol.find(op);
  if (it != opToLineCol.end()) {
    line_num = it->second.first;
  }

  J.object([&] {
    J.attribute("file-line", line_num);
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
                                             const group_type_t &group_type,
                                             local_sec_info_t &sec_info) {
  auto group_info = LocalGenInterface::getGroupInfo(op, n_step, h_step);
  auto in_slice = Slice({{group_info.n_idx, sec_info.n_slice},
                         {-1, -1},
                         {sec_info.h_idx, sec_info.h_slice},
                         {sec_info.w_idx, sec_info.w_slice}});

  std::string layout = group_info.eu_align ? "en_align" : "";

  json::Array operands, results;
  for (auto v : op->getOperands()) {
    if (module::isNone(v)) {
      operands.push_back(json::Object());
    } else
      operands.push_back(record_tensor(v, group_type, in_slice, layout));
  }

  auto out_slice = Slice({{group_info.n_idx, sec_info.out_n_slice},
                          {-1, -1},
                          {sec_info.out_h_idx, sec_info.out_h_slice},
                          {sec_info.out_w_idx, sec_info.out_w_slice}});

  for (auto v : op->getResults()) {
    if (module::isNone(v)) {
      results.push_back(json::Object());
    } else
      results.push_back(record_tensor(v, group_type, out_slice, layout));
  }
  record_loc(op, operands, results);
}

void TensorLocationImpl::after_codegen_global(Operation *op) {
  json::Array operands, results;

  for (auto v : op->getOperands()) {
    if (module::isNone(v)) {
      operands.push_back(json::Object());
    } else
      operands.push_back(record_tensor(v, GROUP_NORMAL, "", ""));
  }

  for (auto v : op->getResults()) {
    if (module::isNone(v)) {
      results.push_back(json::Object());
    } else
      results.push_back(record_tensor(v, GROUP_NORMAL, "", ""));
  }
  record_loc(op, operands, results);
}

} // namespace mlir
