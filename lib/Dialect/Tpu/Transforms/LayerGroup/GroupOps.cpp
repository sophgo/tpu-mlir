//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"
#include <chrono>
#include <fstream>

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/InternalOptimizer.h"
#include "tpu_mlir/Support/TopoSorter.h"

using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

static void
get_ddr_access_statistic_before_layergroup(::mlir::func::FuncOp func) {
  int64_t total_bytes = 0;
  int64_t output_bytes = 0;
  std::ofstream file("group_before.txt", std::ios::ate | std::ios::out);
  func.walk([&](Operation *op) {
    if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
      // do nothing
    } else {
      if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          if (!isa<NoneType>(v.getType())) {
            auto width = module::getDtypeSize(v);
            auto num_elts = module::getNumElements(v);
            output_bytes += num_elts * width;
          }
        }
      }
      for (auto v : op->getOperands()) {
        if (isa<NoneType>(v.getType())) {
          continue;
        }
        if (!isa<tpu::ReshapeOp>(op)) {
          auto width = module::getDtypeSize(v);
          auto num_elts = module::getNumElements(v);
          total_bytes += num_elts * width;
          // file << module::getName(v).str()<<std::endl;
        }
      }
    }
  });
  file.close();
  llvm::errs() << "output_bytes: " << output_bytes
               << ", before group, NetStatisticPass total_bytes: "
               << total_bytes << "\n";
}

void GroupOps::init(LgOptions &options, MLIRContext *ctx) {
  options_ = options;
  ctx_ = ctx;
  MAX_ID_ = llvm::maxIntN(64);
  lg_pass_ir_ = new LgPassIR();
  // lg_pass_ir_->func = std::nullptr;
  version = 0; // 0:old version, 1:new backend api
}

static void collect_ops(SmallVector<Operation *> &ops, LgPassIR *ir) {
  for (auto op : ops) {
    ir->subnet_ops.insert(op);
    for (auto v : op->getOperands()) {
      if (isa<NoneType>(v.getType())) {
        continue;
      }
      ir->subnet_values.insert(v);
    }
    for (auto v : op->getResults()) {
      ir->subnet_values.insert(v);
    }
  }
}

static void collect_ops(FuncOp func, LgPassIR *ir) {
  auto runmode = getRunMode(func);
  std::vector<std::pair<std::string, std::string>> edges;

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
      // do nothing
    } else {
      ir->subnet_ops.insert(op);

      if (runmode == RunMode::TPU_STATIC && !isa<ReturnOp>(op)) {
        auto end = module::getName(op);
        for (auto v : op->getOperands()) {
          if (isa<NoneType>(v.getType())) {
            continue;
          }
          if (isa<BlockArgument>(v)) {
            continue;
          }
          if (v.getDefiningOp() && isa<top::WeightOp>(v.getDefiningOp())) {
            continue;
          }
          auto start = module::getName(v);
          if (start == "image") {
            llvm::errs();
          }
          edges.push_back(std::make_pair(start.str(), end.str()));
        }
      }

      for (auto v : op->getOperands()) {
        if (isa<NoneType>(v.getType())) {
          continue;
        }
        ir->subnet_values.insert(v);
      }
      for (auto v : op->getResults()) {
        bool is_used = false;
        for (auto dst_op : v.getUsers()) {
          if (ir->subnet_ops.contains(dst_op)) {
            is_used = true;
            break;
          }
        }
        if (!is_used) {
          ir->subnet_values.insert(v);
        }
      }
    }
    return WalkResult::advance();
  });

  // ==== do topo sort to build better ordered op list ====
  if (runmode == RunMode::TPU_STATIC) {
    // Has problems, need better toposorter
    // TopoSorter sorter;
    // // 1. sort
    // auto top_order = sorter.topologicalSortWithPriority(edges);

    // // 2. validation and detection
    // bool doReorder = true;
    // if (ir->subnet_ops.size() != (top_order.size() + 1)) {
    //   doReorder = false;
    // } else {

    //   int oriCost = 0;
    //   int time = 0;
    //   std::unordered_map<std::string, int> oriOrder;
    //   for (auto it : llvm::enumerate(ir->subnet_ops)) {
    //     if (!isa<ReturnOp>(it.value())) {
    //       oriOrder[module::getName(it.value()).str()] = it.index();
    //     }
    //   }
    //   for (auto it : llvm::enumerate(ir->subnet_ops)) {
    //     if (!isa<ReturnOp>(it.value())) {
    //       oriCost +=
    //           it.index() -
    //           oriOrder[sorter.getParent(module::getName(it.value()).str())];
    //       time++;
    //     }
    //   }

    //   if (oriCost <= sorter.getCost() || time != sorter.getTime()) {
    //     doReorder = false;
    //   }
    // }

    // // temp close this logic
    // doReorder = false;
    // // 3. do it
    // if (doReorder) {
    //   // adjust ir->subnet_ops
    //   std::vector<Operation *> vector(ir->subnet_ops.size());
    //   for (auto op : ir->subnet_ops) {
    //     if (!isa<ReturnOp>(op)) {
    //       vector[top_order[module::getName(op).str()]] = op;
    //     } else {
    //       vector[ir->subnet_ops.size() - 1] = op;
    //     }
    //   }

    //   // adjust mlir context to avoid "does not dominate this use" problem
    //   ir->subnet_ops.clear();
    //   for (auto it : llvm::enumerate(vector)) {
    //     auto op = it.value();
    //     ir->subnet_ops.insert(op);
    //     if (it.index() >= 1) {
    //       op->moveAfter(vector[it.index() - 1]);
    //       DEBUG_WITH_TYPE("topo_reorder_mlir", {
    //         llvm::dbgs() << "; action = topo"
    //                      << "; before_op = "
    //                      << module::getName(vector[it.index() - 1])
    //                      << "; op = " << module::getName(op) << "\n";
    //       });
    //     } else {
    //       op->moveBefore(vector[it.index() + 1]);
    //       DEBUG_WITH_TYPE("topo_reorder_mlir", {
    //         llvm::dbgs() << "; action = topo"
    //                      << "; before_op = " << module::getName(op)
    //                      << "; op = " << module::getName(vector[it.index() +
    //                      1])
    //                      << "\n";
    //       });
    //     }
    //     // op->dump();
    //     for (auto opd : op->getOperands()) {
    //       auto opdOp = opd.getDefiningOp();
    //       if (opdOp && isa<top::WeightOp>(opdOp)) {
    //         opdOp->moveBefore(op);
    //         DEBUG_WITH_TYPE("topo_reorder_mlir", {
    //           llvm::dbgs() << "; action = topo"
    //                        << "; step = moveWeight"
    //                        << "; weightOp = " << module::getName(opdOp)
    //                        << "; op = " << module::getName(op) << "\n";
    //         });
    //       }
    //     }
    //   }

    //   auto &lastOp = func.getBody().back().back();
    //   if (!isa<ReturnOp>(lastOp)) {
    //     vector.back()->moveAfter(&lastOp);
    //   }
    // }
  }
}

GroupOps::GroupOps(SmallVector<Operation *> &ops, LgOptions &options) {
  init(options, ops[0]->getContext());
  collect_ops(ops, lg_pass_ir_);
}

GroupOps::GroupOps(::mlir::func::FuncOp func, LgOptions &options) {
  init(options, func.getContext());
  collect_ops(func, lg_pass_ir_);
  func_ = func;

  get_ddr_access_statistic_before_layergroup(func);

  if (options_.opt != 3) {
    return;
  }

  Operation *dot_root_op = nullptr;
  std::vector<Operation *> global_layers, tmp_ops, excluded_ops, ops_bk;
  for (auto itr : lg_pass_ir_->subnet_ops) {
    if (!dot_root_op &&
        module::isDebugCmdEnable("dot_root_op_name-" +
                                 module::getName(itr).str() + ",")) {
      llvm::errs() << "GroupOps find dot_root_op_name:"
                   << module::getName(itr).str() << "\n";
      dot_root_op = itr;
    }
    if (module::isDebugCmdEnable("user_defined_global_op-" +
                                 module::getName(itr).str())) {
      global_layers.push_back(itr);
    }
    tmp_ops.push_back(itr);
  }

  if (module::isDebugCmdEnable("dot_root_op_name") && dot_root_op) {
    std::vector<Operation *> op_tree, exclude_ops, break_ops;
    find_op_tree_by_root2(dot_root_op, op_tree, tmp_ops, exclude_ops, break_ops,
                          0, 8);
    auto dot_graph_log_subnet = createSubnetGraph(op_tree);
    dot_graph_log_subnet->export_dot(
        "svg_initial_" + module::getName(module::getModuleOp()).str(), true);
  }

  if (module::isDebugCmdEnable("export_full_svg")) {
    auto dot_graph_log_subnet = createSubnetGraph(tmp_ops);
    dot_graph_log_subnet->export_dot(
        "svg_initial_" + module::getName(module::getModuleOp()).str(), true);
  }

  func.walk([&](Operation *op) {
    if (isa<ReturnOp>(op)) {
      lg_pass_ir_->returnOp = op;
    }
    return WalkResult::advance();
  });

  findSpecialGroup(lg_pass_ir_->subnet_ops);
}

void GroupOps::process() {
  buildGroups();
  if (options_.opt == 3) {
    buildMlir_for_opt3();
  } else {
    buildMlir();
    if (module::isBM1688() && (options_.nnvlc_mode == NnvlcMode::ACTIVATION ||
                               options_.nnvlc_mode == NnvlcMode::ALL)) {
      buildNnvlcActivation();
    }
  }
}

void GroupOps::buildGroups() {
  auto pm = std::make_shared<LgPassManager>();
  auto inner_optimizer = std::make_unique<InternalLgOptimizer>();
  inner_optimizer->manage_passes(pm, options_);
  inner_optimizer->manage_post_passes(pm, options_);
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
      buildGroupOp(lg_infos[i], lg_pass_ir_->shape_secs[i], i);
    }
  }
  // update group overlap info
  int64_t idx = 0;
  for (int64_t i = group_num - 1; i >= 0; --i) {
    if (lg_infos[i].group_ops.size() > 1) {
      time_step = lg_pass_ir_->time_steps[i];
      UpdateGroupOverlapInfo(groups_[idx++], i);
    }
  }

  // update Conv use_3ic_optimze info
  for (auto op : lg_pass_ir_->subnet_ops) {
    if (auto conv_op = dyn_cast<tpu::Conv2DOp>(op)) {
      int use_3ic = conv_op.getUse_3icOptimize();
      Operation *pre_op = conv_op.getInput().getDefiningOp();
      if (use_3ic > 0 && pre_op && !isa<tpu::LoadOp>(pre_op)) {
        // broadcast input using BDC rather than GDMA
        if (!module::isBM1684Family())
          use_3ic |= 0x10;
        conv_op.setUse_3icOptimize(use_3ic);
      }
    }
  }

  DEBUG_WITH_TYPE("dominate_bug", { module::getModuleOp().dump(); });
}

void GroupOps::buildGroupOp(const LgInfo &lg_info,
                            const shape_secs_t &shape_secs,
                            const int64_t group_idx) {
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
  int64_t csecs = shape_secs.csecs;
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
      builder.getNamedAttr("csecs", builder.getI64IntegerAttr(csecs)));
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

      auto cur_ts_tensors = time_step->getTensors(ts);
      for (auto tensor : cur_ts_tensors) {
        if (time_step->get_tensor_swpipl_stage(tensor.first) == stg) {
          if (self_up_overlap_ops.find(tensor.first) !=
              self_up_overlap_ops.end()) {
            self_up_overlap_ops_[group_idx][tensor.first] = id;
          }
          if (self_down_overlap_ops.find(tensor.first) !=
              self_down_overlap_ops.end()) {
            self_down_overlap_ops_[group_idx][tensor.first] = id;
          }
          if (tensor.second.mode == TIMESTEP_LOAD) {
            CreateLoadOp(tensor, id++, ops, lg_info.type);
          } else if (tensor.second.mode == TIMESTEP_STORE) {
            auto storeOp = CreateStoreOp(tensor, id++, lg_info.type);
            stores.push_back(storeOp.getOutput());
          }
        }
      }

      if (stg == 1) {
        auto cur_ts_layers = time_step->getLayers(ts);
        for (auto op : cur_ts_layers) {
          UpdateOpLgParam(op, tensor_infos, id++, lg_info.type);
          if (current_op_ != nullptr) {
            op->moveAfter(current_op_);
          }
          current_op_ = op;
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
      auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                     (int64_t)0, (int64_t)0);
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
      auto ginfo = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0,
                                     (int64_t)0, (int64_t)0);
      flow.push_back(ginfo.id);
    } // cur_tensors
    timestep--;
  }
  groupOp->setAttr("flow", builder.getI64ArrayAttr(flow));
  groups_.push_back(groupOp.getOperation());
}

void GroupOps::UpdateGroupOverlapInfo(Operation *op, int64_t group_idx) {
  auto builder = OpBuilder(ctx_);
  auto groupOp = dyn_cast<tpu::GroupOp>(op);
  auto &self_up_overlap_ops = time_step->get_self_up_overlap_ops();
  auto &self_down_overlap_ops = time_step->get_self_down_overlap_ops();
  auto &other_up_overlap_ops = time_step->get_other_up_overlap_ops();
  auto &other_down_overlap_ops = time_step->get_other_down_overlap_ops();

  // update group_overlap op info of this group
  std::vector<int64_t> self_down_overlap_op;
  for (auto v : self_down_overlap_ops) {
    self_down_overlap_op.push_back(self_down_overlap_ops_[group_idx][v]);
  }
  groupOp->setAttr("self_down_overlap_op",
                   builder.getI64ArrayAttr(self_down_overlap_op));

  std::vector<int64_t> self_up_overlap_op;
  for (auto v : self_up_overlap_ops) {
    self_up_overlap_op.push_back(self_up_overlap_ops_[group_idx][v]);
  }
  groupOp->setAttr("self_up_overlap_op",
                   builder.getI64ArrayAttr(self_up_overlap_op));

  std::vector<int64_t> other_down_overlap_op;
  for (auto &elt : other_down_overlap_ops) {
    other_down_overlap_op.push_back(-(elt.first + 1));
    for (auto v : elt.second) {
      other_down_overlap_op.push_back(self_down_overlap_ops_[group_idx - 1][v]);
    }
  }
  groupOp->setAttr("other_down_overlap_op",
                   builder.getI64ArrayAttr(other_down_overlap_op));

  std::vector<int64_t> other_up_overlap_op;
  for (auto &elt : other_up_overlap_ops) {
    other_up_overlap_op.push_back(-(elt.first + 1));
    for (auto v : elt.second) {
      other_up_overlap_op.push_back(self_up_overlap_ops_[group_idx + 1][v]);
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
  if (auto weightop = dyn_cast_or_null<top::WeightOp>(inputOp)) {
    bool allow_split = false;
    buffer_key.type = LMEM_WEIGHT;
    if (weightop.getAllowSplitAttr() != nullptr) {
      allow_split = true;
    }
    if (allow_split == false) {
      attrs.push_back(builder.getNamedAttr(
          "lmem_type", builder.getI64IntegerAttr(LMEM_WEIGHT)));
    } else {
      attrs.push_back(builder.getNamedAttr(
          "lmem_type", builder.getI64IntegerAttr(LMEM_ACTIVATION)));
    }
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
  auto noneOp = module::getNoneOp(output.getDefiningOp());
  operands.push_back(output);
  operands.push_back(noneOp);
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

LayerGroupAttr
GroupOps::getLgParam(tensor_info_t &tensor_info, int64_t id, int64_t out_addr,
                     int64_t out_size, int64_t group_type, int64_t buffer_addr,
                     int64_t buffer_size, int64_t slice_idx, bool can_merge,
                     std::vector<std::vector<int64_t>> opd_h_slice_offset) {
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
  std::vector<int64_t> c_idxs;
  std::vector<int64_t> c_slices;
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
  for (auto &c : si.c) {
    c_idxs.push_back(c.first);
    c_slices.push_back(c.second);
  }

  std::vector<int64_t> in_hslice_offset;
  for (auto itr : opd_h_slice_offset) {
    for (auto itr2 : itr) {
      in_hslice_offset.push_back(itr2);
    }
    in_hslice_offset.push_back(-1);
  }

  if (buffer_size == 0) {
    buffer_addr = 0;
  }
  return LayerGroupAttr::get(ctx_, out_addr, out_size, buffer_addr, buffer_size,
                             tensor_info.eu_align, can_merge,
                             builder.getDenseI64ArrayAttr(in_hslice_offset),
                             builder.getDenseI64ArrayAttr(n_idxs),
                             builder.getDenseI64ArrayAttr(n_slices),
                             builder.getDenseI64ArrayAttr(c_idxs),
                             builder.getDenseI64ArrayAttr(c_slices),
                             builder.getDenseI64ArrayAttr(d_idxs),
                             builder.getDenseI64ArrayAttr(d_slices),
                             builder.getDenseI64ArrayAttr(h_idxs),
                             builder.getDenseI64ArrayAttr(h_slices),
                             builder.getDenseI64ArrayAttr(w_idxs),
                             builder.getDenseI64ArrayAttr(w_slices), id,
                             tensor_info.stage, slice_idx, group_type);
}

void GroupOps::buildNnvlcActivation() {
  // set compress param for storeop
  for (Operation *op : groups_) {
    if (isa<tpu::GroupOp>(op)) {
      auto groupop = dyn_cast<GroupOp>(op);
      auto &body = groupop.getBody().front();
      body.walk([&](Operation *localop) {
        if (isa<tpu::StoreOp>(localop)) {
          auto dtype = module::getStorageType(localop->getOperand(0));
          if (dtype.isF16() || dtype.isBF16() || dtype.isInteger(8) ||
              dtype.isInteger(16)) {
            bool do_compress = true;
            auto store_result = localop->getResult(0);
            auto yield_op =
                dyn_cast<tpu::YieldOp>(*store_result.getUsers().begin());
            int32_t store_idx;
            for (auto [index, value] :
                 llvm::enumerate(yield_op->getOperands())) {
              if (value == store_result) {
                store_idx = index;
                break;
              }
            }
            auto group_result =
                yield_op->getBlock()->getParentOp()->getResult(store_idx);
            for (auto user : group_result.getUsers()) {
              if (isa<tpu::GroupOp>(user)) {
                for (auto loadop :
                     dyn_cast<GroupOp>(user).getOps<tpu::LoadOp>()) {
                  if (loadop.getOperand() == store_result) {
                    // do not compress if use 3ic_optimize
                    if (loadop->getAttr("use_3ic_optimize")
                            .cast<IntegerAttr>()
                            .getInt() > 0) {
                      do_compress = false;
                      break;
                    }
                  }
                }
              } else if (!user->hasAttr("support_compress") ||
                         (user->hasAttr("support_compress") &&
                          user->getAttr("support_compress")
                                  .cast<BoolAttr>()
                                  .getValue() == false)) {
                do_compress = false;
                break;
              }
            }
            int32_t bias0 = dtype.isInteger(8) ? 127 : 0;
            int32_t bias1 = 0;
            bool is_signed = (dtype.isUnsignedInteger()) ? 0 : 1;
            bool zero_guard = dtype.isInteger(8) ? 0 : 1;
            auto info = CompressAttr::get(op->getContext(), do_compress, false,
                                          bias0, bias1, is_signed, zero_guard);
            localop->setAttr("compress_info", info);
          }
        }
      });
    }
  }

  auto &lg_infos = lg_pass_ir_->lg_infos;
  int64_t group_num = lg_infos.size();
  if (lg_infos.empty()) {
    return;
  }
  // set compress for globalop
  for (int64_t i = group_num - 1; i >= 0; --i) {
    if (lg_infos[i].group_ops.size() < 2) {
      for (auto op : lg_infos[i].group_ops) {
        if (isa<tpu::Conv2DOp>(op)) {
          // set do compress
          auto dtype = module::getStorageType(op->getOperand(0));
          if (dtype.isF16() || dtype.isBF16() || dtype.isInteger(8) ||
              dtype.isInteger(16)) {
            bool do_compress = true;
            for (auto user : op->getUsers()) {
              if (isa<tpu::GroupOp>(user)) {
                auto conv_result = op->getResult(0);
                for (auto loadop :
                     dyn_cast<GroupOp>(user).getOps<tpu::LoadOp>()) {
                  if (loadop.getOperand() == conv_result) {
                    if (loadop->getAttr("use_3ic_optimize")
                            .cast<IntegerAttr>()
                            .getInt() > 0) {
                      do_compress = false;
                      break;
                    }
                  }
                }
              } else if (!user->hasAttr("support_compress") ||
                         (user->hasAttr("support_compress") &&
                          user->getAttr("support_compress")
                                  .cast<BoolAttr>()
                                  .getValue() == false)) {
                do_compress = false;
                break;
              }
            }
            int32_t bias0 = dtype.isInteger(8) ? 127 : 0;
            int32_t bias1 = 0;
            bool is_signed = (dtype.isUnsignedInteger()) ? 0 : 1;
            bool zero_guard = dtype.isInteger(8) ? 0 : 1;
            auto info = CompressAttr::get(op->getContext(), do_compress, false,
                                          bias0, bias1, is_signed, zero_guard);
            op->setAttr("compress_info", info);
          }
        }
      }
    }
  }

  // set decompress param for loadop
  for (Operation *op : groups_) {
    if (isa<tpu::GroupOp>(op)) {
      auto groupop = dyn_cast<GroupOp>(op);
      auto &body = groupop.getBody().front();
      body.walk([&](Operation *localop) {
        auto preop = localop->getOperand(0).getDefiningOp();
        if (isa<tpu::LoadOp>(localop) &&
            !isa<BlockArgument>(op->getOperand(0)) &&
            !isa<top::WeightOp>(preop)) {
          if (preop != nullptr && isa<tpu::GroupOp>(preop)) { // preop is local
            auto pre_value = localop->getOperand(0);
            auto value_idx = module::getIdx(pre_value);
            auto yield_op_ =
                dyn_cast<GroupOp>(preop).getOps<tpu::YieldOp>().begin();
            auto yield_op = *yield_op_;
            auto storeop =
                yield_op->getOperand(value_idx).getDefiningOp<tpu::StoreOp>();
            if (storeop->hasAttr("compress_info")) {
              auto cinfo_pre =
                  storeop->getAttr("compress_info").cast<tpu::CompressAttr>();
              bool do_decompress = cinfo_pre.getDoCompress();
              int32_t bias0 = cinfo_pre.getBias0();
              int32_t bias1 = cinfo_pre.getBias1();
              bool is_signed = cinfo_pre.getIsSigned();
              bool zero_guard = cinfo_pre.getZeroGuard();

              auto info =
                  CompressAttr::get(op->getContext(), false, do_decompress,
                                    bias0, bias1, is_signed, zero_guard);
              localop->setAttr("compress_info", info);
            }
          } else if (preop != nullptr &&
                     preop->hasAttr("compress_info")) { // preop is global
            auto cinfo_pre =
                preop->getAttr("compress_info").cast<tpu::CompressAttr>();
            bool do_decompress = cinfo_pre.getDoCompress();
            int32_t bias0 = cinfo_pre.getBias0();
            int32_t bias1 = cinfo_pre.getBias1();
            bool is_signed = cinfo_pre.getIsSigned();
            bool zero_guard = cinfo_pre.getZeroGuard();

            auto info =
                CompressAttr::get(op->getContext(), false, do_decompress, bias0,
                                  bias1, is_signed, zero_guard);
            localop->setAttr("compress_info", info);
          }
        }
      });
    }
  }

  // set decompress for globalop
  for (int64_t i = group_num - 1; i >= 0; --i) {
    if (lg_infos[i].group_ops.size() < 2) {
      for (auto op : lg_infos[i].group_ops) {
        if (isa<tpu::Conv2DOp>(op) && op->hasAttr("compress_info") &&
            !isa<BlockArgument>(op->getOperand(0)) &&
            !isa<BlockArgument>(op->getOperand(1))) {
          uint32_t idx;
          if (!isa<top::WeightOp>(op->getOperand(0).getDefiningOp()) &&
              !module::getStorageType(op->getOperand(0)).isF32()) {
            idx = 0;
          } else if (!isa<top::WeightOp>(op->getOperand(1).getDefiningOp()) &&
                     !module::getStorageType(op->getOperand(1)).isF32()) {
            idx = 1;
          }
          auto preop = op->getOperand(idx).getDefiningOp();

          bool do_decompress = false;
          if (isa<tpu::GroupOp>(preop)) { // preop is local
            auto pre_value = op->getOperand(idx);
            auto value_idx = module::getIdx(pre_value);
            auto yield_op_ =
                dyn_cast<GroupOp>(preop).getOps<tpu::YieldOp>().begin();
            auto yield_op = *yield_op_;
            auto storeop =
                yield_op->getOperand(value_idx).getDefiningOp<tpu::StoreOp>();
            if (storeop->hasAttr("compress_info")) {
              auto cinfo_pre =
                  storeop->getAttr("compress_info").cast<tpu::CompressAttr>();
              do_decompress = cinfo_pre.getDoCompress();
              bool do_compress = op->getAttr("compress_info")
                                     .cast<tpu::CompressAttr>()
                                     .getDoCompress();
              int32_t bias0 = cinfo_pre.getBias0();
              int32_t bias1 = cinfo_pre.getBias1();
              bool is_signed = cinfo_pre.getIsSigned();
              bool zero_guard = cinfo_pre.getZeroGuard();
              auto info = CompressAttr::get(op->getContext(), do_compress,
                                            do_decompress, bias0, bias1,
                                            is_signed, zero_guard);
              op->setAttr("compress_info", info);
            }
          } else if (preop->hasAttr("compress_info")) { // preop is global
            auto cinfo_pre =
                preop->getAttr("compress_info").cast<tpu::CompressAttr>();
            do_decompress = cinfo_pre.getDoCompress();
            bool do_compress = op->getAttr("compress_info")
                                   .cast<tpu::CompressAttr>()
                                   .getDoCompress();
            int32_t bias0 = cinfo_pre.getBias0();
            int32_t bias1 = cinfo_pre.getBias1();
            bool is_signed = cinfo_pre.getIsSigned();
            bool zero_guard = cinfo_pre.getZeroGuard();
            auto info =
                CompressAttr::get(op->getContext(), do_compress, do_decompress,
                                  bias0, bias1, is_signed, zero_guard);
            op->setAttr("compress_info", info);
          }
        }
      }
    }
  }
}
