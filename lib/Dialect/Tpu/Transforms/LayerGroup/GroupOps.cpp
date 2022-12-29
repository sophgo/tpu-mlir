//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <numeric>

using namespace mlir;
using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

static bool is_ts_overlapped(int64_t start_ts0, int64_t end_ts0,
                             int64_t start_ts1, int64_t end_ts1) {
  bool ret = false;
  if (start_ts0 <= end_ts0) {
    ret = !(
        (start_ts1 <= end_ts1 &&
         (start_ts1 > end_ts0 || end_ts1 < start_ts0)) ||
        (start_ts1 > end_ts1 && (start_ts1 > end_ts0 && end_ts1 < start_ts0)));
  } else {
    ret = !(start_ts1 > end_ts0 && start_ts1 < start_ts0 && end_ts1 > end_ts0 &&
            end_ts1 < start_ts0);
  }
  return ret;
}

lmem_info_t *GroupOps::find_lmem_info(group_lmem_t &group_lmem, mlir::Value v) {
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

lmem_info_t *GroupOps::find_lmem_info(group_lmem_t &group_lmem,
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
  if (isWeightValue(opd)) {
    if (isa<tpu::Conv1DOp, tpu::Conv2DOp, tpu::Conv3DOp, tpu::DeconvOp>(op) &&
        (opd == op->getOperand(1) || opd == op->getOperand(2))) {
      return false;
    }
    if (module::isBM1686()) {
      if (isa<tpu::RequantIntAxisOp>(op) && (opd == op->getOperand(1))) {
        return false;
      }
    }
  }
  return true;
}

bool GroupOps::need_bcast(Value opd) {
  if (opd.hasOneUse() == false) {
    return false;
  }
  auto use_op = *opd.getUsers().begin();
  if (auto cast_op = dyn_cast<tpu::LutOp>(use_op)) {
    return opd == cast_op.table();
  }
  return false;
}

int64_t GroupOps::use_3ic(Value opd) {
  for (auto use_op : opd.getUsers()) {
    if (isa<tpu::GroupOp>(*use_op))
      continue;
    if (auto cast_op = dyn_cast<tpu::Conv2DOp>(*use_op)) {
      if (opd == cast_op.input()) {
        return cast_op.use_3ic_optimize();
      }
    }
  }
  return 0;
}

// assign id for all Value and Op
// mark input and output
group_lmem_t GroupOps::list_lmems(int64_t start_idx, int64_t end_idx) {
  assert(end_idx > start_idx);
  int64_t id = 0;
  auto lmems = std::make_shared<std::vector<lmem_info_t>>();
  for (auto idx = start_idx; idx <= end_idx; idx++) {
    auto op = all_ops[idx];
    if (isa<top::WeightOp>(op)) {
      continue;
    }
    for (auto opd : op->getOperands()) {
      auto in_op_ = opd.getDefiningOp();
      if (in_op_ != nullptr) {
        if (isa<top::NoneOp>(in_op_)) {
          continue;
        }
        if (isa<top::WeightOp>(in_op_)) {
          auto eu_align = is_eu_align(opd, op);
          lmem_info_t li(LMEM_WEIGHT, id, opd, in_op_, eu_align);
          lmems->emplace_back(li);
          id++;
          continue;
        }
      }
      auto it = find_lmem_info(lmems, opd);
      if (it != nullptr) {
        continue;
      }

      lmem_info_t li(LMEM_ACTIVATION, id, opd, in_op_);
      li.is_input = true;
      lmems->emplace_back(li);
      id++;
    }

    lmem_info_t li(LMEM_OPERATION, id, op->getResult(0), op);
    lmems->emplace_back(li);
    id++;
    for (auto out : op->getResults()) {
      lmem_info_t li_out(LMEM_ACTIVATION, id, out, op);
      lmems->emplace_back(li_out);
      id++;
    }
  }
  // mark output
  for (auto &linfo : *lmems) {
    if (linfo.type != LMEM_ACTIVATION || linfo.is_input) {
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
  MAX_ID = llvm::maxIntN(64);
  func = func_;
  ctx = func.getContext();
  func.walk([&](Operation *op) {
    if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
      // do nothing
    } else {
      all_ops.push_back(op);
      auto opds = op->getOperands();
      for (auto v : opds) {
        if (std::find(all_tensors.begin(), all_tensors.end(), v) ==
            all_tensors.end()) {
          all_tensors.push_back(v);
        }
      }
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
      end_idx -= 1;
      continue;
    }
    auto new_start_idx = start_idx;
    auto group_lmem = CreateGroup(start_idx, end_idx, new_start_idx);
    if (group_lmem != nullptr) {
      assert(new_start_idx >= start_idx);
      groups.push_back(group_pair_t(new_start_idx, end_idx));
      groups_ops.push_back(group_ops);
      all_lmems.push_back(group_lmem);
      time_steps.push_back(time_step);
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
  for (uint32_t i = 0; i < all_lmems.size(); ++i) {
    auto &group_lmem = all_lmems[i];
    time_step = time_steps[i];
    group_ops = groups_ops[i];
    buildGroupOp(group_lmem);
  }
}

bool GroupOps::need_none(group_lmem_t &group_lmem) {
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

void GroupOps::buildGroupOp(group_lmem_t &group_lmem) {
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
  attrs.push_back(
      builder.getNamedAttr("swpipl_stage_num", builder.getI64IntegerAttr(3)));
  builder.setInsertionPointAfter(ops.back());
  auto groupOp =
      builder.create<tpu::GroupOp>(func.getLoc(), ret_types, operands, attrs);
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
  llvm::SmallVector<Location, 8> locs;
  int64_t id = 0;
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_OPERATION) {
      auto op = linfo.op;
      UpdateOpLgParam(group_lmem, linfo, id);
      if (current_op != nullptr) {
        op->moveAfter(current_op);
      }
      current_op = op;
      id++;
    } else if (linfo.is_input || linfo.type == LMEM_WEIGHT) {
      CreateLoadOp(linfo, ops, id);
      id++;
    } else if (linfo.is_output) {
      auto storeOp = CreateStoreOp(linfo, id);
      stores.push_back(storeOp.output());
      locs.push_back(storeOp.getLoc());
      id++;
    }
  }
  auto group_loc = builder.getFusedLoc(locs);
  builder.setInsertionPointAfter(current_op);
  builder.create<tpu::YieldOp>(group_loc, stores);

  // update flow attribute
  std::vector<int64_t> flow;
  int64_t timestep = -1;
  for (uint32_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
    flow.push_back(timestep);
    auto cur_ops = time_step->getOps(ts);
    for (auto op : cur_ops) {
      auto lgOp = dyn_cast<LocalGenInterface>(op);
      auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0);
      flow.push_back(ginfo.id);
    } // cur_ops
    auto cur_tensors = time_step->getValues(ts);
    for (auto v : cur_tensors) {
      auto op = v.getDefiningOp();
      if (op != nullptr && !isa<top::WeightOp>(op) &&
          std::find(group_ops->begin(), group_ops->end(), op) !=
              group_ops->end()) {
        for (auto user : v.getUsers()) {
          if (isa<tpu::StoreOp>(user)) {
            op = user;
            break;
          }
        }
      } else {
        for (auto user : v.getUsers()) {
          if (isa<tpu::LoadOp>(user)) {
            op = user;
            break;
          }
        }
      }
      auto lgOp = dyn_cast<LocalGenInterface>(op);
      auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0);
      flow.push_back(ginfo.id);
    } // cur_tensors
    timestep--;
  }

  groupOp->setAttr("flow", builder.getI64ArrayAttr(flow));
  groupOp->setLoc(group_loc);
}

void GroupOps::UpdateOpLgParam(group_lmem_t &group_lmem, lmem_info_t &linfo,
                               int64_t id) {
  assert(linfo.type == LMEM_OPERATION);
  auto op = linfo.op;
  auto output = find_lmem_info(group_lmem, linfo.value);
  op->setAttr(LocalGenInterface::kLayerGroupAttrName,
              getLgParam(*output, id, linfo.stage, linfo.addr, linfo.size));
}

void GroupOps::CreateLoadOp(lmem_info_t &linfo,
                            const std::vector<mlir::Operation *> &ops,
                            int64_t id) {
  auto builder = OpBuilder(ctx);
  auto input = linfo.value;
  auto inputOp = input.getDefiningOp();
  std::vector<Value> operands;
  operands.push_back(input);
  std::vector<NamedAttribute> attrs;
  std::string name = "load_";
  if (inputOp != nullptr) {
    name = name + module::getName(inputOp).str();
  } else {
    name = name + std::to_string(input.cast<BlockArgument>().getArgNumber());
  }
  if (need_bcast(input)) {
    attrs.push_back(
        builder.getNamedAttr("do_bcast", builder.getBoolAttr(true)));
  }
  if (auto use_3ic_optimize = use_3ic(input)) {
    attrs.push_back(builder.getNamedAttr(
        "use_3ic_optimize", builder.getI64IntegerAttr(use_3ic_optimize)));
  }
  attrs.push_back(builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                                       getLgParam(linfo, id, linfo.stage)));
  if (current_op == nullptr) {
    builder.setInsertionPointToStart(body);
  } else if (isa<tpu::StoreOp>(current_op)) {
    builder.setInsertionPoint(current_op);
  } else {
    builder.setInsertionPointAfter(current_op);
  }
  auto loadOp =
      builder.create<tpu::LoadOp>(NameLoc::get(builder.getStringAttr(name)),
                                  input.getType(), operands, attrs);
  input.replaceUsesWithIf(loadOp.output(), [&](OpOperand &operand) {
    Operation *user = operand.getOwner();
    return find(ops.begin(), ops.end(), user) != ops.end();
  });
  if (current_op == nullptr || !isa<tpu::StoreOp>(current_op)) {
    current_op = loadOp;
  }
}

StoreOp GroupOps::CreateStoreOp(lmem_info_t &linfo, int64_t id) {
  auto builder = OpBuilder(ctx);
  auto output = linfo.value;
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  operands.push_back(output);
  std::string name = module::getName(output).str();
  attrs.push_back(builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                                       getLgParam(linfo, id, linfo.stage)));
  builder.setInsertionPointAfter(current_op);
  auto storeOp =
      builder.create<tpu::StoreOp>(NameLoc::get(builder.getStringAttr(name)),
                                   output.getType(), operands, attrs);
  current_op = storeOp;
  return storeOp;
}

LayerGroupAttr GroupOps::getLgParam(lmem_info_t &linfo, int64_t id,
                                    int64_t stage, int64_t buffer_addr,
                                    int64_t buffer_size) {
  auto builder = OpBuilder(ctx);
  auto &si = linfo.slice_info;
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
  return LayerGroupAttr::get(ctx, linfo.addr, linfo.size, buffer_addr,
                             buffer_size, linfo.eu_align, h_idxs, h_slices,
                             n_idxs, n_slices, id, stage);
}

group_lmem_t GroupOps::CreateGroup(int64_t start_idx, int64_t end_idx,
                                   int64_t &new_start_idx) {
  if (check_group(start_idx, end_idx) == false) {
    return nullptr;
  }
  int nsecs = 1, hsecs = 1;
  // try no slice first
  // new_start_idx = start_idx;
  // while (new_start_idx < end_idx) {
  //   auto group_lmem = CreateGroupBySecs(new_start_idx, end_idx, nsecs,
  //   hsecs); if (group_lmem) {
  //     return group_lmem;
  //   }
  //   new_start_idx++;
  // }
  auto end_op = all_ops[end_idx];
  auto out = end_op->getResult(0);
  int64_t n, c, h, w;
  module::getNCHW(out, n, c, h, w);
  auto type = module::getStorageType(out);
  auto n_align = Arch::get_n_align(type.getIntOrFloatBitWidth() / 8);
  int64_t max_nsecs = ceiling_func(n, n_align);
  int64_t max_hsecs = h;
  new_start_idx = start_idx;
  while (end_idx > new_start_idx) {
    no_more_try_secs = false;
    // slice n first
    for (nsecs = 1, hsecs = 1; no_more_try_secs == false && nsecs <= max_nsecs;
         nsecs++) {
      auto group_lmem = CreateGroupBySecs(new_start_idx, end_idx, nsecs, hsecs);
      if (group_lmem) {
        return group_lmem;
      }
    }
    // slice h
    nsecs = max_nsecs;
    for (hsecs = 1; no_more_try_secs == false && hsecs <= max_hsecs; hsecs++) {
      auto group_lmem = CreateGroupBySecs(new_start_idx, end_idx, nsecs, hsecs);
      if (group_lmem) {
        return group_lmem;
      }
    }
    new_start_idx++;
  }
  return nullptr;
}

bool GroupOps::slice_all_outputs(group_lmem_t &group_lmem, int64_t nsecs,
                                 int64_t hsecs) {
  for (auto &linfo : *group_lmem) {
    if (linfo.is_output == false) {
      continue;
    }
    int64_t n, c, h, w;
    module::getNCHW(linfo.value, n, c, h, w);
    slice_pair_t slice_pair;
    auto &si = linfo.slice_info;
    if (nsecs == 1) {
      si.n.emplace_back(slice_pair_t(0, n));
    } else {
      auto type = module::getStorageType(linfo.value);
      auto n_align = Arch::get_n_align(type.getIntOrFloatBitWidth() / 8);
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
    } else if (hsecs > h) {
      no_more_try_secs = true;
      return false;
    } else {
      auto per_slice = h / hsecs;
      std::vector<int64_t> slice_v(hsecs, per_slice);
      auto left = h - per_slice * hsecs;
      for (int64_t i = 0; i < left; i++) {
        slice_v[i]++;
      }
      auto offset = 0l;
      for (auto i = 0; i < hsecs; i++) {
        si.h.emplace_back(slice_pair_t(offset, slice_v[i]));
        offset += slice_v[i];
      }
    }
    auto op_info = find_lmem_info(group_lmem, linfo.op);
    assert(op_info != nullptr);
    op_info->slice_info = si;
  }
  return true;
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

bool GroupOps::backward_from_tensor(group_lmem_t &group_lmem,
                                    lmem_info_t *linfo) {
  assert(linfo != nullptr);
  assert(linfo->type == LMEM_ACTIVATION);
  if (linfo->is_input) {
    return true;
  }
  auto &si = linfo->slice_info;
  assert(!si.n.empty());
  assert(!si.h.empty());
  // make sure all users ready
  for (auto user : linfo->value.getUsers()) {
    auto uinfo = find_lmem_info(group_lmem, user);
    if (uinfo == nullptr) {
      continue;
    }
    if (uinfo->op_slice_done == false) {
      return true;
    }
  }
  auto op = linfo->value.getDefiningOp();
  auto lg_op = cast<LocalGenInterface>(op);
  auto op_info = find_lmem_info(group_lmem, op);
  assert(op_info != nullptr);
  if (op_info->op_slice_done) {
    return true;
  }
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
    if (failed(ret) || h_slice == 0) {
      return false;
    }
    slice_h.emplace_back(slice_pair_t(h_idx, h_slice));
  }
  llvm::SmallVector<lmem_info_t *, 8> back_info;
  for (auto opd : op->getOperands()) {
    auto in_info = find_lmem_info(group_lmem, opd);
    if (in_info == nullptr || in_info->type != LMEM_ACTIVATION) {
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
    back_info.push_back(in_info);
  }
  op_info->op_slice_done = true;
  for (auto in_info : back_info) {
    auto ret = backward_from_tensor(group_lmem, in_info);
    if (ret == false) {
      return false;
    }
  }
  return true;
}

bool GroupOps::backward_entry(group_lmem_t &group_lmem) {
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

void GroupOps::set_lmem_size(group_lmem_t &group_lmem) {
  // set size
  int64_t slice_n, slice_h;
  for (auto &linfo : *group_lmem) {
    if (LMEM_WEIGHT == linfo.type) {
      linfo.size = Arch::get_weight_lmem_bytes(linfo.value, linfo.eu_align);
    } else if (LMEM_ACTIVATION == linfo.type) {
      get_max_slice_nh(linfo, slice_n, slice_h);
      linfo.size = Arch::get_tensor_lmem_bytes(linfo.value, slice_n, slice_h,
                                               linfo.eu_align);
    }
  }
  for (auto &linfo : *group_lmem) {
    if (LMEM_OPERATION == linfo.type) {
      auto lg_op = cast<LocalGenInterface>(linfo.op);
      auto in = linfo.op->getOperand(0);
      auto out = linfo.op->getResult(0);
      auto in_info = find_lmem_info(group_lmem, in);
      assert(in_info != nullptr);
      auto out_info = find_lmem_info(group_lmem, out);
      assert(out_info != nullptr);
      int64_t in_nslice, in_hslice, out_nslice, out_hslice;
      get_max_slice_nh(*in_info, in_nslice, in_hslice);
      get_max_slice_nh(*out_info, out_nslice, out_hslice);
      linfo.size = lg_op.getBufferSize(in_info->size, out_info->size, in_nslice,
                                       in_hslice, out_nslice, out_hslice);
    }
  }
}

void GroupOps::assign_timestep(group_lmem_t &group_lmem) {
  int64_t timestep = 0;
  lmem_type_t last_type = LMEM_ANY;
  bool last_output = false;
  int64_t idx = 0;
  std::vector<Operation *> ops_v;
  for (auto &linfo : *group_lmem) {
    switch (linfo.type) {
    case LMEM_OPERATION:
      ops_v.push_back(linfo.op);
      if (last_type != LMEM_OPERATION) {
        timestep++;
        last_type = linfo.type;
      }
      break;
    case LMEM_WEIGHT:
      if (last_output) {
        timestep--;
      }
      last_type = linfo.type;
      break;
    case LMEM_ACTIVATION:
      if (linfo.is_input) {
        if (last_output) {
          timestep--;
        }
        last_type = linfo.type;
      } else if (linfo.is_output && last_type == LMEM_OPERATION) {
        timestep++;
        last_type = linfo.type;
      }
      break;
    default:
      break;
    }
    last_output = linfo.is_output;
    linfo.timestep = timestep;
    linfo.start_timestep = 0;
    linfo.end_timestep = 0;
    idx++;
  }

  // create timestep table
  time_step = std::make_shared<BasicTimeStep>();
  int64_t ts = 0;
  TpuTsField tpu_field;
  GdmaTsField gdma_field;
  while (ts <= timestep) {
    tpu_field.clear();
    gdma_field.clear();
    for (auto &linfo : *group_lmem) {
      if (linfo.timestep != ts) {
        continue;
      }
      if (linfo.type == LMEM_OPERATION) {
        tpu_field.emplace_back(linfo.op);
      } else if (linfo.is_input || linfo.is_output ||
                 linfo.type == LMEM_WEIGHT) {
        gdma_field.emplace_back(linfo.value);
      }
    }
    time_step->add_tpu0_gdma0_ts_field(tpu_field, gdma_field);
    ts++;
  }
  time_step->software_pipeline();

  // map op to timestep
  std::map<Operation *, int64_t, op_compare> op_timestep;
  for (auto op : ops_v) {
    for (uint32_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
      auto ops = time_step->getOps(ts);
      if (std::find(ops.begin(), ops.end(), op) != ops.end()) {
        op_timestep.insert(std::pair<Operation *, int64_t>(op, ts));
        break;
      }
    }
  }

  // map value to timestep
  std::map<Value, int64_t, value_compare> tensor_timestep;
  for (auto &linfo : *group_lmem) {
    if (linfo.type != LMEM_OPERATION) {
      for (uint32_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
        auto tensors = time_step->getValues(ts);
        if (std::find(tensors.begin(), tensors.end(), linfo.value) !=
            tensors.end()) {
          tensor_timestep.insert(std::pair<Value, int64_t>(linfo.value, ts));
          break;
        }
      }
      if (linfo.type == LMEM_ACTIVATION && !linfo.is_input &&
          !linfo.is_output) {
        for (uint32_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
          auto src_op = linfo.value.getDefiningOp();
          auto ops = time_step->getOps(ts);
          if (std::find(ops.begin(), ops.end(), src_op) != ops.end()) {
            tensor_timestep.insert(std::pair<Value, int64_t>(linfo.value, ts));
            break;
          }
        }
      }
    }
  }

  // update timestep and stage of op/value in group_lmem
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_WEIGHT || linfo.type == LMEM_ACTIVATION) {
      linfo.stage = time_step->get_tensor_swpipl_stage(linfo.value);
      linfo.timestep = tensor_timestep[linfo.value];
      linfo.start_timestep = linfo.timestep;
      if (linfo.is_output) {
        linfo.end_timestep = linfo.timestep;
      }
      if (linfo.type == LMEM_ACTIVATION && !linfo.is_input) {
        auto op = linfo.value.getDefiningOp();
        if (op_timestep.find(op) != op_timestep.end()) {
          linfo.start_timestep = op_timestep[op];
        }
      }
      auto users = linfo.value.getUsers();
      for (auto op : users) {
        if (op_timestep.find(op) != op_timestep.end()) {
          linfo.end_timestep = std::max(op_timestep[op], linfo.end_timestep);
        }
      }
    }
    if (linfo.type == LMEM_OPERATION) {
      linfo.stage = time_step->get_layer_swpipl_stage(linfo.op);
      linfo.timestep = op_timestep[linfo.op];
      linfo.start_timestep = op_timestep[linfo.op];
      linfo.end_timestep = op_timestep[linfo.op];
    }
  }
}

void GroupOps::adjust_lmem_id(group_lmem_t &group_lmem, int64_t nsecs,
                              int64_t hsecs) {
  bool no_slice = (nsecs == 1 && hsecs == 1);
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_WEIGHT && no_slice == false) {
      linfo.hold_in_lmem = true;
    }
  }
}

void GroupOps::check_group_lmem(group_lmem_t &group_lmem,
                                int64_t timestep_num) {
  for (auto &linfo : *group_lmem) {
    if (linfo.type != LMEM_OPERATION) {
      llvm::errs() << "==== id = " << linfo.id << ", ";
      linfo.value.print(llvm::errs());
      llvm::errs() << ": start_ts = " << linfo.start_timestep
                   << ", end_ts = " << linfo.end_timestep << "\n";
    }
  }
  std::list<std::pair<addr_pair_t, int64_t>> ts_lmems; // ((addr, size), id)
  for (int64_t ts = 0; ts < timestep_num; ++ts) {
    ts_lmems.clear();
    for (auto &linfo : *group_lmem) {
      if (linfo.addr == -1 || linfo.size == 0) {
        continue;
      }
      if (linfo.hold_in_lmem ||
          is_ts_overlapped(ts, ts, linfo.start_timestep, linfo.end_timestep)) {
        auto it = ts_lmems.begin();
        for (; it != ts_lmems.end(); ++it) {
          if (linfo.addr < it->first.first) {
            break;
          }
        }
        auto pair =
            std::make_pair(addr_pair_t(linfo.addr, linfo.size), linfo.id);
        ts_lmems.insert(it, pair);
      }
    }
    llvm::errs() << "=============Timestep idx = " << ts << ":==============\n";
    int64_t end_addr = 0;
    for (auto it : ts_lmems) {
      if (end_addr > it.first.first) {
        llvm::errs()
            << "++++++++++++ lmem is overlapped here +++++++++++++++++++++++++";
      }
      end_addr = it.first.first + it.first.second;
      llvm::errs() << it.first.first << "\t" << it.first.second << "\t"
                   << end_addr << "\t" << it.second << "\n";
    }
  }
}

group_lmem_t GroupOps::CreateGroupBySecs(int64_t start_idx, int64_t end_idx,
                                         int64_t nsecs, int64_t hsecs) {
  auto group_lmem = list_lmems(start_idx, end_idx);
  auto ret = slice_all_outputs(group_lmem, nsecs, hsecs);
  if (ret == false) {
    return nullptr;
  }
  ret = backward_entry(group_lmem);
  if (ret == false) {
    return nullptr;
  }
  // checkout all values have been sliced
  for (auto &linfo : *group_lmem) {
    if (linfo.type != LMEM_ACTIVATION) {
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

  group_ops = std::make_shared<std::vector<Operation *>>();
  for (auto &linfo : *group_lmem) {
    if (linfo.type == LMEM_OPERATION) {
      group_ops->push_back(linfo.op);
    }
  }

  return std::move(group_lmem);
}

lmem_info_t *GroupOps::find_max_unalloc_lmem(group_lmem_t &group_lmem,
                                             int64_t ts, lmem_type_t type) {
  int64_t size = 0;
  int64_t term = -1;
  lmem_info_t *p_li = nullptr;
  int timestep_num = time_step->get_timestep_num();
  for (auto &linfo : *group_lmem) {
    if (linfo.addr != -1 || linfo.size == 0) {
      continue;
    }
    if (ts != -1) {
      if ((linfo.stage == 1 &&
           (linfo.start_timestep > ts || linfo.end_timestep < ts)) ||
          (linfo.stage != 1 &&
           (linfo.start_timestep > ts && linfo.end_timestep < ts))) {
        continue;
      }
    }
    if (type != LMEM_ANY) {
      if (type != linfo.type) {
        continue;
      }
    }
    auto term_ = linfo.end_timestep - linfo.start_timestep;
    if (linfo.end_timestep < linfo.start_timestep) {
      term_ = linfo.end_timestep + timestep_num - linfo.start_timestep;
    }
    if (term_ > term || (term_ == term && linfo.size > size)) {
      term = term_;
      size = linfo.size;
      p_li = &linfo;
    }
  }
  return p_li;
}

bool GroupOps::assign_lmem_addr(group_lmem_t &group_lmem, int64_t nsecs,
                                int64_t hsecs) {
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

  lmem_info_t *p_li = nullptr;
  do {
    p_li = find_max_unalloc_lmem(group_lmem);
    if (p_li != nullptr) {
      rebuild_alloc_lmem(group_lmem, p_li->start_timestep, p_li->end_timestep);
      p_li->addr = alloc_lmem(p_li->size);
      if (p_li->addr < 0) {
        return false;
      }
    }
  } while (p_li != nullptr);
  return true;
}

int64_t GroupOps::alloc_lmem(int64_t size) {
  bool in_bank[] = {true, false};
  if (size > Arch::LMEM_BYTES) {
    return -1;
  }
  for (auto align_bank : in_bank) {
    if (align_bank && size > Arch::LMEM_BANK_BYTES) {
      continue;
    }
    int64_t addr = 0;
    for (auto it = allocated_lmems.begin(); it != allocated_lmems.end(); ++it) {
      if (addr + size <= it->first) {
        auto pair = addr_pair_t(addr, size);
        allocated_lmems.insert(it, pair);
        return addr;
      }
      int64_t addr_tmp = align_up(it->first + it->second, Arch::EU_BYTES);
      if (align_bank) {
        auto bank0 = ceiling_func(addr_tmp, Arch::LMEM_BANK_BYTES);
        auto bank1 = ceiling_func(addr_tmp + size - 1, Arch::LMEM_BANK_BYTES);
        if (bank0 != bank1) {
          addr_tmp = align_up(addr_tmp, Arch::LMEM_BANK_BYTES);
        }
      }
      addr = std::max(addr, addr_tmp);
    }
    if (addr + size > Arch::LMEM_BYTES) {
      continue;
    }
    auto pair = addr_pair_t(addr, size);
    allocated_lmems.push_back(pair);
    return addr;
  }
  return -1;
}

void GroupOps::rebuild_alloc_lmem(group_lmem_t &group_lmem, int64_t start_ts,
                                  int64_t end_ts) {
  allocated_lmems.clear();
  for (auto &linfo : *group_lmem) {
    if (linfo.addr == -1 || linfo.size == 0) {
      continue;
    }
    if (linfo.hold_in_lmem ||
        is_ts_overlapped(start_ts, end_ts, linfo.start_timestep,
                         linfo.end_timestep)) {
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
  assert(lmem_info.type == LMEM_ACTIVATION);
  auto &si_h = lmem_info.slice_info.h;
  assert(lmem_info.slice_info.h.size() > 0);
  int64_t n, c, h, w;
  module::getNCHW(lmem_info.value, n, c, h, w);
  int64_t total_h = 0;
  for (auto &it : si_h) {
    total_h += it.second;
  }
  if (total_h * 2 > h * 3) { // h increase 1.5 times
    return false;
  }
  return true;
}
