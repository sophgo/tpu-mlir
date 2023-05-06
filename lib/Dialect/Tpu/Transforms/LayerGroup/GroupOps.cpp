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

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/InternalOptimizer.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LgPass.h"


using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

GroupOps::GroupOps(::mlir::func::FuncOp func) {
  MAX_ID_ = llvm::maxIntN(64);
  func_ = func;
  ctx_ = func.getContext();
  lg_pass_ir_ = new LgPassIR();

  func.walk([&](Operation *op) {
    if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
      // do nothing
    } else {
      lg_pass_ir_->subnet_ops.push_back(op);
      for (auto v : op->getOperands()) {
        if (v.getType().isa<NoneType>()) {
          continue;
        }
        if (std::find(lg_pass_ir_->subnet_values.begin(),
                      lg_pass_ir_->subnet_values.end(),
                      v) == lg_pass_ir_->subnet_values.end()) {
          lg_pass_ir_->subnet_values.push_back(v);
        }
      }
      for (auto v : op->getResults()) {
        bool is_used = false;
        for (auto dst_op : v.getUsers()) {
          if (std::find(lg_pass_ir_->subnet_ops.begin(),
                        lg_pass_ir_->subnet_ops.end(),
                        dst_op) != lg_pass_ir_->subnet_ops.end()) {
            is_used = true;
            break;
          }
        }
        if (!is_used) {
          lg_pass_ir_->subnet_values.push_back(v);
        }
      }
    }
  });
}

void GroupOps::process(int64_t opt) {
  buildGroups(opt);
  buildMlir();
}

void GroupOps::buildGroups(int64_t opt) {
  LgOptions options;
  options.dyn_compile = false;
  options.opt = opt;
  auto pm = std::make_shared<LgPassManager>();
  auto inner_optimizer = std::make_unique<InternalLgOptimizer>();
  inner_optimizer->manage_passes(pm, options);
  inner_optimizer->manage_post_passes(pm, options);
  pm->run(lg_pass_ir_);
}

void GroupOps::buildMlir() {
  auto &lg_infos = lg_pass_ir_->lg_infos;
  if (lg_infos.empty()) {
    return;
  }

  groups_.clear();
  self_up_overlap_ops_.clear();
  self_down_overlap_ops_.clear();
  int64_t group_num = lg_infos.size();
  for (int64_t i = group_num - 1; i >= 0; --i) {
    if (lg_infos[i].group_ops.size() > 1) {
      time_step = lg_pass_ir_->time_steps[i];
      buildGroupOp(lg_infos[i], lg_pass_ir_->shape_secs[i]);
    }
  }
  // update group overlap info
  int64_t idx = 0;
  for (int64_t i = group_num - 1; i >= 0; --i) {
    if (lg_infos[i].group_ops.size() > 1) {
      time_step = lg_pass_ir_->time_steps[i];
      UpdateGroupOverlapInfo(groups_[idx++]);
    }
  }
}

void GroupOps::buildGroupOp(const LgInfo &lg_info,
                            const shape_secs_t &shape_secs) {
  auto builder = OpBuilder(ctx_);
  llvm::SmallVector<Value, 8> operands;
  llvm::SmallVector<Value, 8> outputs;
  llvm::SmallVector<NamedAttribute, 8> attrs;
  llvm::SmallVector<Type, 8> in_types;
  // llvm::SmallVector<Location, 8> in_locs;
  llvm::SmallVector<Type, 8> ret_types;
  auto &ops = lg_info.group_ops;
  auto &tensor_infos = time_step->get_tensor_infos();

  int64_t nsecs = shape_secs.nsecs;
  int64_t hsecs = shape_secs.hsecs;
  int64_t dsecs = shape_secs.dsecs;
  int64_t wsecs = shape_secs.wsecs;
  for (auto in : lg_info.group_ins) {
    in_types.push_back(in.getType());
    // in_locs.push_back(module::getLoc(in));
    operands.push_back(in);
  }
  llvm::SmallVector<Location, 8> locs;
  for (auto out : lg_info.group_outs) {
    ret_types.push_back(out.getType());
    locs.push_back(module::getLoc(out));
    outputs.push_back(out);
  }
  auto group_loc = builder.getFusedLoc(locs);
  attrs.push_back(
      builder.getNamedAttr("nsecs", builder.getI64IntegerAttr(nsecs)));
  attrs.push_back(
      builder.getNamedAttr("hsecs", builder.getI64IntegerAttr(hsecs)));
  attrs.push_back(
      builder.getNamedAttr("dsecs", builder.getI64IntegerAttr(dsecs)));
  attrs.push_back(
      builder.getNamedAttr("wsecs", builder.getI64IntegerAttr(wsecs)));
  attrs.push_back(
      builder.getNamedAttr("swpipl_stage_num", builder.getI64IntegerAttr(3)));
  attrs.push_back(builder.getNamedAttr(
      "group_type", builder.getI64IntegerAttr((int64_t)lg_info.type)));
  builder.setInsertionPointAfter(ops.back());
  auto groupOp =
      builder.create<tpu::GroupOp>(group_loc, ret_types, operands, attrs);
  body_ = new Block();
  groupOp.getBody().push_back(body_);

  //  replace outputs
  for (auto it : llvm::enumerate(groupOp.getResults())) {
    outputs[it.index()].replaceUsesWithIf(it.value(), [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return find(ops.begin(), ops.end(), user) == ops.end();
    });
  }

  // record current group overlap_ops and its op id in GroupOp
  auto &self_up_overlap_ops = time_step->get_self_up_overlap_ops();
  auto &self_down_overlap_ops = time_step->get_self_down_overlap_ops();

  current_op_ = nullptr;
  llvm::SmallVector<Value, 8> stores;
  int64_t id = 0;
  for (int64_t stg = 0; stg < 3; ++stg) {
    for (size_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
      if (stg == 1) {
        auto cur_ts_layers = time_step->getLayers(ts);
        for (auto op : cur_ts_layers) {
          UpdateOpLgParam(op, tensor_infos, id++, lg_info.type);
          op->moveAfter(current_op_);
          current_op_ = op;
        }
      }

      auto cur_ts_tensors = time_step->getTensors(ts);
      for (auto tensor : cur_ts_tensors) {
        if (time_step->get_tensor_swpipl_stage(tensor.first) == stg) {
          if (self_up_overlap_ops.find(tensor.first) !=
              self_up_overlap_ops.end()) {
            self_up_overlap_ops_[tensor.first] = id;
          }
          if (self_down_overlap_ops.find(tensor.first) !=
              self_down_overlap_ops.end()) {
            self_down_overlap_ops_[tensor.first] = id;
          }
          if (tensor.second.mode == TIMESTEP_LOAD) {
            CreateLoadOp(tensor, id++, ops, lg_info.type);
          } else if (tensor.second.mode == TIMESTEP_STORE) {
            auto storeOp = CreateStoreOp(tensor, id++, lg_info.type);
            stores.push_back(storeOp.getOutput());
          }
        }
      }
    }
  }
  builder.setInsertionPointAfter(current_op_);
  // reorder stores
  llvm::SmallVector<Value, 8> new_stores;
  for (auto &loc : locs) {
    for (auto &s : stores) {
      if (module::getLoc(s) == loc) {
        new_stores.push_back(s);
        break;
      }
    }
  }
  assert(new_stores.size() == stores.size());
  builder.create<tpu::YieldOp>(group_loc, new_stores);

  // update flow attribute
  std::vector<int64_t> flow;
  int64_t timestep = -1;
  for (uint32_t ts = 0; ts < time_step->get_timestep_num(); ++ts) {
    flow.push_back(timestep);
    auto cur_ops = time_step->getLayers(ts);
    for (auto op : cur_ops) {
      auto lgOp = dyn_cast<LocalGenInterface>(op);
      auto ginfo =
          lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0, (int64_t)0);
      flow.push_back(ginfo.id);
    } // cur_ops
    auto cur_tensors = time_step->getTensors(ts);
    for (auto iter : cur_tensors) {
      auto v = iter.first;
      auto op = v.getDefiningOp();
      if (op != nullptr && !isa<top::WeightOp>(op) &&
          std::find(ops.begin(), ops.end(), op) != ops.end()) {
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
      auto ginfo =
          lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0, (int64_t)0);
      flow.push_back(ginfo.id);
    } // cur_tensors
    timestep--;
  }
  groupOp->setAttr("flow", builder.getI64ArrayAttr(flow));
  groups_.push_back(groupOp.getOperation());
}

void GroupOps::UpdateGroupOverlapInfo(Operation *op) {
  auto builder = OpBuilder(ctx_);
  auto groupOp = dyn_cast<tpu::GroupOp>(op);
  auto &self_up_overlap_ops = time_step->get_self_up_overlap_ops();
  auto &self_down_overlap_ops = time_step->get_self_down_overlap_ops();
  auto &other_up_overlap_ops = time_step->get_other_up_overlap_ops();
  auto &other_down_overlap_ops = time_step->get_other_down_overlap_ops();

  // update group_overlap op info of this group
  std::vector<int64_t> self_down_overlap_op;
  for (auto v : self_down_overlap_ops) {
    self_down_overlap_op.push_back(self_down_overlap_ops_[v]);
  }
  groupOp->setAttr("self_down_overlap_op",
                   builder.getI64ArrayAttr(self_down_overlap_op));

  std::vector<int64_t> self_up_overlap_op;
  for (auto v : self_up_overlap_ops) {
    self_up_overlap_op.push_back(self_up_overlap_ops_[v]);
  }
  groupOp->setAttr("self_up_overlap_op",
                   builder.getI64ArrayAttr(self_up_overlap_op));

  std::vector<int64_t> other_down_overlap_op;
  for (auto &elt : other_down_overlap_ops) {
    other_down_overlap_op.push_back(-(elt.first + 1));
    for (auto v : elt.second) {
      other_down_overlap_op.push_back(self_down_overlap_ops_[v]);
    }
  }
  groupOp->setAttr("other_down_overlap_op",
                   builder.getI64ArrayAttr(other_down_overlap_op));

  std::vector<int64_t> other_up_overlap_op;
  for (auto &elt : other_up_overlap_ops) {
    other_up_overlap_op.push_back(-(elt.first + 1));
    for (auto v : elt.second) {
      other_up_overlap_op.push_back(self_up_overlap_ops_[v]);
    }
  }
  groupOp->setAttr("other_up_overlap_op",
                   builder.getI64ArrayAttr(other_up_overlap_op));
}

void GroupOps::CreateLoadOp(GdmaElt &tensor, int64_t id,
                            const std::vector<Operation *> &ops,
                            group_type_t group_type) {
  auto builder = OpBuilder(ctx_);
  auto input = tensor.first;
  auto ti = tensor.second;
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
  if (ti.need_bcast) {
    attrs.push_back(
        builder.getNamedAttr("do_bcast", builder.getBoolAttr(true)));
  }
  if (ti.use_3ic_opt) {
    attrs.push_back(builder.getNamedAttr(
        "use_3ic_optimize", builder.getI64IntegerAttr(ti.use_3ic_opt)));
  }

  mem_buffer_key_t buffer_key = {LMEM_ACTIVATION, input, nullptr};
  if (dyn_cast_or_null<top::WeightOp>(inputOp)) {
    buffer_key.type = LMEM_WEIGHT;
    attrs.push_back(builder.getNamedAttr(
        "lmem_type", builder.getI64IntegerAttr(LMEM_WEIGHT)));
  } else {
    attrs.push_back(builder.getNamedAttr(
        "lmem_type", builder.getI64IntegerAttr(LMEM_ACTIVATION)));
  }
  tpu_mlir::tpu::mem_buffer_value_t buffer_value;
  if (module::isBM1684Family() && module::isWeight(tensor.first) &&
      llvm::any_of(tensor.first.getUsers(),
                   [](Operation *op) { return isa<tpu::LutOp>(op); })) {
    buffer_value = time_step->get_l2mem_buffer_value(buffer_key);
  } else {
    buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  }
  // auto &buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  attrs.push_back(builder.getNamedAttr(
      LocalGenInterface::kLayerGroupAttrName,
      getLgParam(ti, id, buffer_value.addr, buffer_value.size, group_type)));
  if (current_op_ == nullptr) {
    builder.setInsertionPointToStart(body_);
  } else if (isa<tpu::StoreOp>(current_op_)) {
    builder.setInsertionPoint(current_op_);
  } else {
    builder.setInsertionPointAfter(current_op_);
  }
  auto loadOp =
      builder.create<tpu::LoadOp>(NameLoc::get(builder.getStringAttr(name)),
                                  input.getType(), operands, attrs);
  input.replaceUsesWithIf(loadOp.getOutput(), [&](OpOperand &operand) {
    Operation *user = operand.getOwner();
    return std::find(ops.begin(), ops.end(), user) != ops.end();
  });
  if (current_op_ == nullptr || !isa<tpu::StoreOp>(current_op_)) {
    current_op_ = loadOp;
  }
}

StoreOp GroupOps::CreateStoreOp(GdmaElt &tensor, int64_t id,
                                group_type_t group_type) {
  auto builder = OpBuilder(ctx_);
  auto output = tensor.first;
  auto &ti = tensor.second;
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  operands.push_back(output);
  std::string name = module::getName(output).str();

  mem_buffer_key_t buffer_key = {LMEM_ACTIVATION, output, nullptr};
  auto &buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  attrs.push_back(builder.getNamedAttr(
      LocalGenInterface::kLayerGroupAttrName,
      getLgParam(ti, id, buffer_value.addr, buffer_value.size, group_type)));
  builder.setInsertionPointAfter(current_op_);
  auto storeOp =
      builder.create<tpu::StoreOp>(NameLoc::get(builder.getStringAttr(name)),
                                   output.getType(), operands, attrs);
  current_op_ = storeOp;
  return storeOp;
}

void GroupOps::UpdateOpLgParam(Operation *op, TensorInfo &tensor_infos,
                               int64_t id, group_type_t group_type) {
  auto output = *op->getResults().begin();
  auto &ti = tensor_infos[output];
  ti.stage = 1;
  mem_buffer_key_t buffer_key = {LMEM_OPERATION, output, op};
  auto &imm_buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  buffer_key.type = LMEM_ACTIVATION;
  auto &out_buffer_value = time_step->get_lmem_buffer_value(buffer_key);
  op->setAttr(LocalGenInterface::kLayerGroupAttrName,
              getLgParam(ti, (int64_t)id, out_buffer_value.addr,
                         out_buffer_value.size, group_type,
                         imm_buffer_value.addr, imm_buffer_value.size));
}

LayerGroupAttr GroupOps::getLgParam(tensor_info_t &tensor_info, int64_t id,
                                    int64_t out_addr, int64_t out_size,
                                    int64_t group_type, int64_t buffer_addr,
                                    int64_t buffer_size) {
  auto builder = OpBuilder(ctx_);
  auto &si = tensor_info.slice_info;
  std::vector<int64_t> h_idxs;
  std::vector<int64_t> h_slices;
  std::vector<int64_t> n_idxs;
  std::vector<int64_t> n_slices;
  std::vector<int64_t> d_idxs;
  std::vector<int64_t> d_slices;
  std::vector<int64_t> w_idxs;
  std::vector<int64_t> w_slices;
  for (auto &h : si.h) {
    h_idxs.push_back(h.first);
    h_slices.push_back(h.second);
  }
  for (auto &n : si.n) {
    n_idxs.push_back(n.first);
    n_slices.push_back(n.second);
  }
  for (auto &d : si.d) {
    d_idxs.push_back(d.first);
    d_slices.push_back(d.second);
  }
  for (auto &w : si.w) {
    w_idxs.push_back(w.first);
    w_slices.push_back(w.second);
  }

  if (buffer_size == 0) {
    buffer_addr = 0;
  }
  return LayerGroupAttr::get(ctx_, out_addr, out_size, buffer_addr, buffer_size,
                             tensor_info.eu_align,
                             builder.getDenseI64ArrayAttr(n_idxs),
                             builder.getDenseI64ArrayAttr(n_slices),
                             builder.getDenseI64ArrayAttr(d_idxs),
                             builder.getDenseI64ArrayAttr(d_slices),
                             builder.getDenseI64ArrayAttr(h_idxs),
                             builder.getDenseI64ArrayAttr(h_slices),
                             builder.getDenseI64ArrayAttr(w_idxs),
                             builder.getDenseI64ArrayAttr(w_slices), id,
                             tensor_info.stage, group_type);
}
