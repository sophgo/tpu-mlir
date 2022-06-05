//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "GroupOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <numeric>

using namespace mlir;
using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

lmem_info_t *GroupOps::find_lmem_info(group_lmem_t group_lmem, mlir::Value v) {
  if (group_lmem == nullptr || v == nullptr) {
    return nullptr;
  }
  for (auto &it : *group_lmem) {
    if (it.type != LMEM_OPERATION && it.value == v) {
      return &it;
    }
  }
  return nullptr;
}

lmem_info_t *GroupOps::find_lmem_info(group_lmem_t group_lmem,
                                      mlir::Operation *op) {
  if (group_lmem == nullptr || op == nullptr) {
    return nullptr;
  }
  for (auto &it : *group_lmem) {
    if (it.type == LMEM_OPERATION && it.op == op) {
      return &it;
    }
  }
  return nullptr;
}

bool GroupOps::isWeightValue(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<top::WeightOp>(op)) {
    return true;
  }
  return false;
}

bool GroupOps::is_eu_align(Value opd, Operation *op) {
  if (auto castOp = dyn_cast<tpu::ConvOp>(op)) {
    if (opd == castOp.filter() || opd == castOp.bias()) {
      return false;
    }
  }
  return true;
}

group_lmem_t GroupOps::list_lmems(int64_t start_idx, int64_t end_idx) {
  assert(end_idx > start_idx);
  int64_t id = 0;
  auto lmems = std::make_shared<std::vector<lmem_info_t>>();
  for (auto idx = start_idx; idx <= end_idx; idx++) {
    auto op = all_ops[idx];
    if (isa<top::WeightOp>(op)) {
      continue;
    }
    int64_t op_id = id + op->getNumOperands();
    for (auto opd : op->getOperands()) {
      auto in_op_ = opd.getDefiningOp();
      if (in_op_ != nullptr) {
        if (isa<top::NoneOp>(in_op_)) {
          continue;
        }
        if (isa<top::WeightOp>(in_op_)) {
          auto eu_align = is_eu_align(opd, op);
          lmem_info_t li(LMEM_WEIGHT, id, op_id, op_id, opd, in_op_, eu_align);
          lmems->emplace_back(li);
          id++;
          continue;
        }
      }
      auto it = find_lmem_info(lmems, opd);
      if (it != nullptr) {
        it->end_id = op_id;
        continue;
      }

      lmem_info_t li(LMEM_TENSOR, id, op_id, op_id, opd, in_op_);
      li.is_input = true;
      lmems->emplace_back(li);
      id++;
    }
    auto out = op->getResult(0);
    lmem_info_t li(LMEM_OPERATION, op_id, op_id, op_id, out, op);
    lmems->emplace_back(li);
    id = op_id + 1;
    lmem_info_t li_out(LMEM_TENSOR, id, op_id, op_id, out, op);
    lmems->emplace_back(li_out);
    id++;
  }
  // mark output
  for (auto &linfo : *lmems) {
    if (linfo.type != LMEM_TENSOR || linfo.is_input) {
      continue;
    }
    for (auto user : linfo.value.getUsers()) {
      if (find_lmem_info(lmems, user) == nullptr) {
        linfo.is_output = true;
        break;
      }
    }
  }
  return std::move(lmems);
}

GroupOps::GroupOps(::mlir::func::FuncOp func_) {
  int id = 1;
  MAX_ID = llvm::maxIntN(64);
  func = func_;
  ctx = func.getContext();
  auto chip = Module::getChip(Module::getModuleOp(func.getOperation()));
  bm168x = BM168x::instance(chip);
  n_align = bm168x->get_n_align(1);
  func.walk([&](Operation *op) {
    if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
      // do nothing
    } else {
      all_ops.push_back(op);
    }
  });
}

bool GroupOps::isLgSupport(int64_t op_idx) {
  int64_t num_ops = all_ops.size();
  assert(op_idx < num_ops);
  auto op = all_ops[op_idx];
  if (isa<top::WeightOp>(op)) {
    return true;
  }
  auto lg_if = dyn_cast<tpu_mlir::LocalGenInterface>(op);
  if (!lg_if || mlir::failed(lg_if.LocalGenSupport())) {
    return false;
  }
  return true;
}

void GroupOps::buildGroups() {
  int64_t num_ops = all_ops.size();
  int64_t start_idx = 0, end_idx = num_ops - 1;
  // from end to start search
  while (end_idx >= 0) {
    while (end_idx >= 0) {
      if (isLgSupport(end_idx) == true) {
        break;
      }
      end_idx--;
    }
    start_idx = end_idx;
    while (start_idx > 0) {
      if (isLgSupport(start_idx - 1) == false) {
        break;
      }
      start_idx--;
    }
    if (start_idx == end_idx) {
      end_idx -= 2;
      continue;
    }
    auto new_start_idx = start_idx;
    auto group_lmem = CreateGroup(start_idx, end_idx, new_start_idx);
    if (group_lmem != nullptr) {
      assert(new_start_idx >= start_idx);
      groups.push_back(group_pair_t(new_start_idx, end_idx));
      all_lmems.push_back(group_lmem);
      end_idx = new_start_idx - 1;
      continue;
    }
    end_idx--;
  }
}

bool GroupOps::check_group(int64_t start_idx, int64_t end_idx) {
  if (start_idx >= end_idx) {
    return false;
  }
  std::set<Operation *> ops;
  for (auto i = start_idx; i <= end_idx; i++) {
    ops.insert(all_ops[i]);
  }
  for (auto op : ops) {
    // is all input
    bool is_all_input = true;
    for (auto opd : op->getOperands()) {
      auto in_ = opd.getDefiningOp();
      if (in_ == nullptr) {
        continue;
      }
      if (isa<top::NoneOp, top::WeightOp>(in_)) {
        continue;
      }
      if (ops.find(in_) != ops.end()) {
        is_all_input = false;
        break;
      }
    }
    if (is_all_input == false) {
      continue;
    }
    bool is_all_output = true;
    for (auto user : op->getUsers()) {
      if (ops.find(user) != ops.end()) {
        is_all_output = false;
        break;
      }
    }
    if (is_all_output == false) {
      continue;
    }
    return false;
  }
  return true;
}

void GroupOps::process() {
  buildGroups();
  // dump all groups
  llvm::errs() << "dump all groups: \n";
  for (auto g : groups) {
    llvm::errs() << "[" << g.first << ", " << g.second << "]\n";
  }
  buildMlir();
}

void GroupOps::buildMlir() {
  if (groups.empty()) {
    return;
  }
  auto num_groups = groups.size();
  assert(num_groups == all_lmems.size());
  for (auto group_lmem : all_lmems) {
    buildGroupOp(group_lmem);
  }
}

bool GroupOps::need_none(group_lmem_t group_lmem) {
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_OPERATION) {
      for (auto opd : linfo.op->getOperands()) {
        if (opd.getType().isa<NoneType>()) {
          return true;
        }
      }
    }
  }
  return false;
}

void GroupOps::buildGroupOp(group_lmem_t group_lmem) {
  auto builder = OpBuilder(ctx);
  llvm::SmallVector<Value, 8> operands;
  llvm::SmallVector<Value, 8> outputs;
  llvm::SmallVector<NamedAttribute, 8> attrs;
  llvm::SmallVector<Type, 8> in_types;
  llvm::SmallVector<Location, 8> in_locs;
  llvm::SmallVector<Type, 8> ret_types;
  std::vector<Operation *> ops;
  int64_t nsecs = group_lmem->back().slice_info.n.size();
  int64_t hsecs = group_lmem->back().slice_info.h.size();
  for (auto &linfo : *group_lmem) {
    if (linfo.is_input) {
      in_types.push_back(linfo.value.getType());
      in_locs.push_back(linfo.value.getLoc());
      operands.push_back(linfo.value);
    } else if (linfo.is_output) {
      ret_types.push_back(linfo.value.getType());
      outputs.push_back(linfo.value);
    } else if (linfo.type == LMEM_OPERATION) {
      ops.push_back(linfo.op);
    }
  }
  attrs.push_back(
      builder.getNamedAttr("nsecs", builder.getI64IntegerAttr(nsecs)));
  attrs.push_back(
      builder.getNamedAttr("hsecs", builder.getI64IntegerAttr(hsecs)));
  builder.setInsertionPointAfter(ops.back());
  auto groupOp = builder.create<tpu::GroupOp>(func.getLoc(), ret_types,
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  body = new Block();
  groupOp.body().push_back(body);
  //  replace outputs
  for (auto it : llvm::enumerate(groupOp.getResults())) {
    outputs[it.index()].replaceUsesWithIf(it.value(), [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return find(ops.begin(), ops.end(), user) == ops.end();
    });
  }

  current_op = nullptr;
  llvm::SmallVector<Value, 8> stores;
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_OPERATION) {
      auto op = linfo.op;
      UpdateOpLgParam(group_lmem, linfo);
      if (current_op != nullptr) {
        op->moveAfter(current_op);
      }
      current_op = op;
    } else if (linfo.is_input || linfo.type == LMEM_WEIGHT) {
      CreateLoadOp(linfo, ops);
    } else if (linfo.is_output) {
      auto op = CreateStoreOp(linfo);
      stores.push_back(op->getResult(0));
    }
  }
  builder.setInsertionPointAfter(current_op);
  builder.create<tpu::YieldOp>(func.getLoc(), stores);
}

void GroupOps::UpdateOpLgParam(group_lmem_t group_lmem, lmem_info_t &linfo) {
  assert(linfo.type == LMEM_OPERATION);
  auto op = linfo.op;
  auto output = find_lmem_info(group_lmem, linfo.value);
  op->setAttr(LocalGenInterface::kLayerGroupAttrName,
              getLgParam(*output, linfo.timestep, linfo.addr, linfo.size));
}

void GroupOps::CreateLoadOp(lmem_info_t &linfo,
                            const std::vector<mlir::Operation *> &ops) {
  auto builder = OpBuilder(ctx);
  auto input = linfo.value;
  auto inputOp = input.getDefiningOp();
  std::vector<Value> operands;
  operands.push_back(input);
  std::vector<NamedAttribute> attrs;
  std::string name = "load_";
  if (inputOp != nullptr) {
    name = name + Module::getName(inputOp).str();
  } else {
    name = name + std::to_string(input.cast<BlockArgument>().getArgNumber());
  }
  attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
  attrs.push_back(builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                                       getLgParam(linfo, linfo.timestep)));
  if (current_op != nullptr) {
    builder.setInsertionPointAfter(current_op);
  } else {
    builder.setInsertionPointToStart(body);
  }
  auto loadOp = builder.create<tpu::LoadOp>(func.getLoc(), input.getType(),
                                            ArrayRef<Value>{operands},
                                            ArrayRef<NamedAttribute>{attrs});
  input.replaceUsesWithIf(loadOp.output(), [&](OpOperand &operand) {
    Operation *user = operand.getOwner();
    return find(ops.begin(), ops.end(), user) != ops.end();
  });
  current_op = loadOp;
}

StoreOp GroupOps::CreateStoreOp(lmem_info_t &linfo) {
  auto builder = OpBuilder(ctx);
  auto output = linfo.value;
  auto outputOp = output.getDefiningOp();
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  operands.push_back(output);
  std::string name = "store_" + Module::getName(outputOp).str();
  attrs.push_back(builder.getNamedAttr("name", builder.getStringAttr(name)));
  attrs.push_back(builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                                       getLgParam(linfo, linfo.timestep)));
  builder.setInsertionPointAfter(current_op);
  auto storeOp = builder.create<tpu::StoreOp>(func.getLoc(), output.getType(),
                                              ArrayRef<Value>{operands},
                                              ArrayRef<NamedAttribute>{attrs});
  current_op = storeOp;
  return storeOp;
}

LayerGroup GroupOps::getLgParam(lmem_info_t &linfo, int64_t timestep,
                                int64_t buffer_addr, int64_t buffer_size) {
  auto builder = OpBuilder(ctx);
  auto si = linfo.slice_info;
  std::vector<int64_t> h_idxs;
  std::vector<int64_t> h_slices;
  std::vector<int64_t> n_idxs;
  std::vector<int64_t> n_slices;
  for (auto &h : si.h) {
    h_idxs.push_back(h.first);
    h_slices.push_back(h.second);
  }
  for (auto &n : si.n) {
    n_idxs.push_back(n.first);
    n_slices.push_back(n.second);
  }
  if (buffer_size == 0) {
    buffer_addr = 0;
  }
  return LayerGroup::get(
      builder.getI64IntegerAttr(linfo.addr),
      builder.getI64IntegerAttr(linfo.size),
      builder.getI64IntegerAttr(buffer_addr),
      builder.getI64IntegerAttr(buffer_size),
      builder.getBoolAttr(linfo.eu_align), builder.getI64ArrayAttr(h_idxs),
      builder.getI64ArrayAttr(h_slices), builder.getI64ArrayAttr(n_idxs),
      builder.getI64ArrayAttr(n_slices), builder.getI64IntegerAttr(timestep),
      ctx);
}

group_lmem_t GroupOps::CreateGroup(int64_t start_idx, int64_t end_idx,
                                   int64_t &new_start_idx) {
  if (check_group(start_idx, end_idx) == false) {
    return nullptr;
  }
  int nsecs = 1, hsecs = 1;
  // try no slice first
  new_start_idx = start_idx;
  while (new_start_idx < end_idx) {
    auto group_lmem = CreateGroupBySecs(new_start_idx, end_idx, nsecs, hsecs);
    if (group_lmem) {
      return group_lmem;
    }
    new_start_idx++;
  }
  auto end_op = all_ops[end_idx];
  auto out = end_op->getResult(0);
  int64_t n, c, h, w;
  Module::getNCHW(out, n, c, h, w);
  auto n_align = bm168x->get_n_align(1);
  int64_t max_nsecs = ceiling_func(n, n_align);
  int64_t max_hsecs = h;
  new_start_idx = start_idx;
  while (end_idx > new_start_idx) {
    no_more_try_secs = false;
    hsecs = 1;
    for (nsecs = 2; no_more_try_secs == false && nsecs <= max_nsecs; nsecs++) {
      auto group_lmem = CreateGroupBySecs(new_start_idx, end_idx, nsecs, hsecs);
      if (group_lmem) {
        return group_lmem;
      }
    }

    nsecs = max_nsecs;
    for (hsecs = 2; no_more_try_secs == false && hsecs <= max_hsecs; hsecs++) {
      auto group_lmem = CreateGroupBySecs(new_start_idx, end_idx, nsecs, hsecs);
      if (group_lmem) {
        return group_lmem;
      }
    }
    new_start_idx++;
  }
  return nullptr;
}

void GroupOps::slice_all_outputs(group_lmem_t group_lmem, int64_t nsecs,
                                 int64_t hsecs) {
  for (auto &linfo : *group_lmem) {
    if (linfo.is_output == false) {
      continue;
    }
    int64_t n, c, h, w;
    Module::getNCHW(linfo.value, n, c, h, w);
    slice_pair_t slice_pair;
    auto &si = linfo.slice_info;
    if (nsecs == 1) {
      si.n.emplace_back(slice_pair_t(0, n));
    } else {
      auto max_slice = align_up(ceiling_func(n, nsecs), n_align);
      auto offset = 0l;
      for (auto i = 0; i < nsecs; i++) {
        slice_pair.first = offset;
        slice_pair.second = std::min(max_slice, n - offset);
        si.n.emplace_back(slice_pair);
        offset += slice_pair.second;
      }
    }
    if (hsecs == 1) {
      si.h.emplace_back(slice_pair_t(0, h));
    } else {
      auto max_slice = ceiling_func(h, hsecs);
      auto offset = 0l;
      for (auto i = 0; i < hsecs; i++) {
        slice_pair.first = offset;
        slice_pair.second = std::min(max_slice, h - offset);
        si.h.emplace_back(slice_pair);
        offset += slice_pair.second;
      }
    }
    auto op_info = find_lmem_info(group_lmem, linfo.op);
    assert(op_info != nullptr);
    op_info->slice_info = si;
  }
}

bool GroupOps::is_same_slice(const std::vector<slice_pair_t> &a,
                             const std::vector<slice_pair_t> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (auto it : llvm::zip(a, b)) {
    if (!is_same_slice(std::get<0>(it), std::get<1>(it))) {
      return false;
    }
  }
  return true;
}

bool GroupOps::backward_from_tensor(group_lmem_t group_lmem,
                                    lmem_info_t *linfo) {
  assert(linfo != nullptr);
  assert(linfo->type == LMEM_TENSOR);
  if (linfo->is_input) {
    return true;
  }
  auto &si = linfo->slice_info;
  assert(!si.n.empty());
  assert(!si.h.empty());
  // make sure all users ready
  for (auto user : linfo->value.getUsers()) {
    auto uinfo = find_lmem_info(group_lmem, user->getResult(0));
    if (uinfo == nullptr) {
      continue;
    }
    if (uinfo->slice_info.n.size() == 0) {
      return true;
    }
  }
  auto op = linfo->value.getDefiningOp();
  auto lg_op = cast<LocalGenInterface>(op);
  std::vector<slice_pair_t> slice_n;
  std::vector<slice_pair_t> slice_h;
  for (auto &s : si.n) {
    int64_t n_idx = 0, n_slice = 0;
    auto ret = lg_op.BackwardN(n_idx, n_slice, s.first, s.second);
    if (failed(ret)) {
      return false;
    }
    slice_n.emplace_back(slice_pair_t(n_idx, n_slice));
  }
  for (auto &s : si.h) {
    int64_t h_idx = 0, h_slice = 0;
    auto ret = lg_op.BackwardH(h_idx, h_slice, s.first, s.second);
    if (failed(ret)) {
      return false;
    }
    slice_h.emplace_back(slice_pair_t(h_idx, h_slice));
  }
  for (auto opd : op->getOperands()) {
    auto in_info = find_lmem_info(group_lmem, opd);
    if (in_info == nullptr || in_info->type != LMEM_TENSOR) {
      continue;
    }
    auto si = &in_info->slice_info;
    if (si->n.size() == 0) {
      si->n = slice_n;
      si->h = slice_h;
    } else {
      if (false == is_same_slice(si->n, slice_n) ||
          false == is_same_slice(si->h, slice_h)) {
        return false;
      }
    }
    if (false == check_hsecs(*in_info)) {
      no_more_try_secs = true;
      return false;
    }
    auto ret = backward_from_tensor(group_lmem, in_info);
    if (ret == false) {
      return false;
    }
  }
  return true;
}

bool GroupOps::backward_entry(group_lmem_t group_lmem) {
  for (auto &linfo : *group_lmem) {
    if (linfo.is_output == false) {
      continue;
    }
    auto ret = backward_from_tensor(group_lmem, &linfo);
    if (ret == false) {
      return false;
    }
  }
  return true;
}

void GroupOps::get_max_slice_nh(const lmem_info_t &lmem_info, int64_t &max_n,
                                int64_t &max_h) {
  auto &si = lmem_info.slice_info;
  max_n = 0;
  max_h = 0;
  for (auto &slice : si.n) {
    if (slice.second > max_n) {
      max_n = slice.second;
    }
  }
  for (auto &slice : si.h) {
    if (slice.second > max_h) {
      max_h = slice.second;
    }
  }
}

void GroupOps::set_lmem_size(group_lmem_t group_lmem) {
  // set size
  int64_t n, c, h, w, slice_n, slice_h;
  for (auto &linfo : *group_lmem) {
    if (LMEM_WEIGHT == linfo.type) {
      linfo.size = bm168x->get_weight_lmem_bytes(linfo.value, linfo.eu_align);
    } else if (LMEM_TENSOR == linfo.type) {
      get_max_slice_nh(linfo, slice_n, slice_h);
      linfo.size = bm168x->get_tensor_lmem_bytes(linfo.value, slice_n, slice_h,
                                                 linfo.eu_align);
    }
  }
  for (auto &linfo : *group_lmem) {
    if (LMEM_OPERATION == linfo.type) {
      auto lg_op = cast<LocalGenInterface>(linfo.op);
      auto out = linfo.op->getResult(0);
      Module::getNCHW(out, n, c, h, w);
      auto out_info = find_lmem_info(group_lmem, out);
      assert(out_info != nullptr);
      get_max_slice_nh(*out_info, slice_n, slice_h);
      linfo.size = lg_op.getBufferSize(slice_n, c, slice_h, w, out_info->size);
    }
  }
}

void GroupOps::assign_timestep(group_lmem_t group_lmem) {
  int64_t timestep = 0;
  for (auto &linfo : *group_lmem) {
    switch (linfo.type) {
    case LMEM_OPERATION:
      timestep++;
      break;
    case LMEM_TENSOR:
      if (linfo.is_output) {
        timestep++;
      }
      break;
    default:
      break;
    }
    linfo.timestep = timestep;
  }
}

void GroupOps::adjust_lmem_id(group_lmem_t group_lmem, int64_t nsecs,
                              int64_t hsecs) {
  bool no_slice = (nsecs == 1 && hsecs == 1);
  std::vector<int64_t> ops_v;
  int idx = 0;
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_WEIGHT && no_slice == false) {
      // ajust weight end
      linfo.end_id = MAX_ID;
    }
    if (linfo.type == LMEM_OPERATION) {
      // get op pos
      ops_v.push_back(idx);
    }
    idx++;
  }
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_WEIGHT || linfo.is_input) {
      for (auto id : ops_v) {
        auto &op_info = group_lmem->at(id);
        if (op_info.timestep == linfo.timestep && op_info.id < linfo.start_id) {
          linfo.start_id = op_info.id;
          break;
        }
      }
    }
  }
}

group_lmem_t GroupOps::CreateGroupBySecs(int64_t start_idx, int64_t end_idx,
                                         int64_t nsecs, int64_t hsecs) {
  std::vector<Operation *> group_ops;
  for (auto i = start_idx; i <= end_idx; i++) {
    group_ops.push_back(all_ops[i]);
  }
  auto group_lmem = list_lmems(start_idx, end_idx);
  slice_all_outputs(group_lmem, nsecs, hsecs);
  auto ret = backward_entry(group_lmem);
  if (ret == false) {
    return nullptr;
  }
  // checkout all values have been sliced
  for (auto &linfo : *group_lmem) {
    if (linfo.type != LMEM_TENSOR) {
      continue;
    }
    auto si = linfo.slice_info;
    assert(nsecs == si.n.size());
    assert(hsecs == si.h.size());
  }
  // update timestep
  assign_timestep(group_lmem);
  adjust_lmem_id(group_lmem, nsecs, hsecs);

  // update lmem size and addr
  set_lmem_size(group_lmem);
  ret = assign_lmem_addr(group_lmem, nsecs, hsecs);
  if (ret == false) {
    return nullptr;
  }
  return std::move(group_lmem);
}

lmem_info_t *GroupOps::find_max_unalloc_lmem(group_lmem_t group_lmem,
                                             int64_t op_id, lmem_type_t type) {
  int64_t size = 0;
  int64_t term = -1;
  lmem_info_t *p_li = nullptr;
  for (auto &linfo : *group_lmem) {
    if (linfo.addr != -1 || linfo.size == 0) {
      continue;
    }
    if (op_id >= 0) {
      if (linfo.start_id > op_id || linfo.end_id < op_id) {
        continue;
      }
    }
    if (type != LMEM_ANY) {
      if (type != linfo.type) {
        continue;
      }
    }
    auto term_ = linfo.end_id - linfo.start_id;
    if (term_ > term || (term_ == term && linfo.size > size)) {
      term = term_;
      size = linfo.size;
      p_li = &linfo;
    }
  }
  return p_li;
}

bool GroupOps::assign_lmem_addr(group_lmem_t group_lmem, int64_t nsecs,
                                int64_t hsecs) {
  int64_t start_addr = 0, end_addr = bm168x->get_lmem_bytes();
  allocated_lmems.clear();
  if (nsecs != 1 || hsecs != 1) {
    // weight first
    lmem_info_t *p_li = nullptr;
    do {
      p_li = find_max_unalloc_lmem(group_lmem, -1, LMEM_WEIGHT);
      if (p_li != nullptr) {
        p_li->addr = alloc_lmem(p_li->size);
        if (p_li->addr < 0) {
          no_more_try_secs = true;
          return false;
        }
      }
    } while (p_li != nullptr);
  }
  std::vector<int64_t> op_ids;
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_OPERATION) {
      op_ids.push_back(linfo.id);
    }
  }
  for (auto op_id : op_ids) {
    rebuild_alloc_lmem(group_lmem, op_id);
    lmem_info_t *p_li = nullptr;
    do {
      p_li = find_max_unalloc_lmem(group_lmem, op_id);
      if (p_li != nullptr) {
        p_li->addr = alloc_lmem(p_li->size);
        if (p_li->addr < 0) {
          return false;
        }
      }
    } while (p_li != nullptr);
  }
  return true;
}

int64_t GroupOps::alloc_lmem(int64_t size) {
  bool in_bank[] = {true, false};
  if (size > bm168x->get_lmem_bytes()) {
    return -1;
  }
  for (auto align_bank : in_bank) {
    if (align_bank && size > bm168x->get_lmem_bank_bytes()) {
      continue;
    }
    int64_t addr = 0;
    for (auto it = allocated_lmems.begin(); it != allocated_lmems.end(); ++it) {
      if (addr + size <= it->first) {
        auto pair = addr_pair_t(addr, size);
        allocated_lmems.insert(it, pair);
        return addr;
      }
      addr = align_up(it->first + it->second, bm168x->get_eu_bytes());
      if (align_bank) {
        auto bank0 = ceiling_func(addr, bm168x->get_lmem_bank_bytes());
        auto bank1 =
            ceiling_func(addr + size - 1, bm168x->get_lmem_bank_bytes());
        if (bank0 != bank1) {
          addr = align_up(addr, bm168x->get_lmem_bank_bytes());
        }
      }
    }
    if (addr + size > bm168x->get_lmem_bytes()) {
      continue;
    }
    auto pair = addr_pair_t(addr, size);
    allocated_lmems.push_back(pair);
    return addr;
  }
  return -1;
}

void GroupOps::rebuild_alloc_lmem(group_lmem_t group_lmem, int64_t op_id) {
  allocated_lmems.clear();
  for (auto &linfo : *group_lmem) {
    if (linfo.start_id <= op_id && linfo.end_id >= op_id) {
      if (linfo.addr == -1 || linfo.size == 0) {
        continue;
      }
      auto it = allocated_lmems.begin();
      for (; it != allocated_lmems.end(); ++it) {
        if (linfo.addr < it->first) {
          break;
        }
      }
      auto pair = addr_pair_t(linfo.addr, linfo.size);
      allocated_lmems.insert(it, pair);
    }
  }
}

bool GroupOps::check_hsecs(lmem_info_t &lmem_info) {
  assert(lmem_info.type == LMEM_TENSOR);
  auto &si_h = lmem_info.slice_info.h;
  assert(lmem_info.slice_info.h.size() > 0);
  int64_t n, c, h, w;
  Module::getNCHW(lmem_info.value, n, c, h, w);
  int64_t total_h = 0;
  for (auto &it : si_h) {
    total_h += it.second;
  }
  if (total_h * 2 > h * 3) { // h increase 1.5 times
    return false;
  }
  return true;
}
