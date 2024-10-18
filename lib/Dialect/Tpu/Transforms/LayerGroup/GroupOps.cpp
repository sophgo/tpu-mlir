//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/GroupOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupUtil.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/InternalOptimizer.h"
#include "tpu_mlir/Support/TopoSorter.h"

using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;


void backward_collect_op(const Value &out, std::list<Value> &tensor_branchs, std::vector<Operation*>& last_op_all_pre_ops) {
  auto op = out.getDefiningOp();
  if (op) {
    for (auto in : op->getOperands()) {
      auto pre_op = in.getDefiningOp();
      if (pre_op != nullptr && isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
        continue;
      }
      last_op_all_pre_ops.push_back(pre_op);
      tensor_branchs.push_back(in);
    }
  }
}

void GroupOps::find_local_layer_base_group(Operation * op) {
  tmp_local_layer_group.push_back(op);
  all_local_layer_nodes.push_back(op);

  int i = 0;
  for (auto v : op->getOperands()) {
    auto pre_op = v.getDefiningOp();
    if (pre_op == nullptr || isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
      continue;
    }
    auto itr = std::find(all_local_layer_nodes.begin(), all_local_layer_nodes.end(), pre_op);
    if (itr == all_local_layer_nodes.end() && isLgSupport(pre_op) && isPreOpHaveAComputeOp(pre_op)) { //isPreOpHaveAComputeOp ?? todo
      llvm::outs()<<"i:"<<i<<", op:"<<module::getName(op).str()<<", pre_op:"<<module::getName(pre_op).str()<<"\n";
      find_local_layer_base_group(pre_op);
    }
    i++;
  }

  i = 0;
  for (auto user : op->getUsers()) {
    auto itr = std::find(all_local_layer_nodes.begin(), all_local_layer_nodes.end(), user);
    if (itr == all_local_layer_nodes.end() && isLgSupport(user)) {
      llvm::outs()<<"i:"<<i<<", op:"<<module::getName(op).str()<<", user:"<<module::getName(user).str()<<"\n";
      find_local_layer_base_group(user);
    }
    i++;
  }
}

GroupOps::GroupOps(::mlir::func::FuncOp func, int64_t opt) {
  MAX_ID_ = llvm::maxIntN(64);
  func_ = func;
  ctx_ = func.getContext();
  lg_pass_ir_ = new LgPassIR();
  version = 0; //0:old version, 1:new backend api
  lg_pass_ir_->func = func;

  auto runmode = getRunMode(func);
  std::vector<std::pair<std::string, std::string>> edges;

  func.walk([&](Operation *op) {
    if (isa<FuncOp, top::NoneOp, top::WeightOp>(op)) {
      // do nothing
    } else {
      lg_pass_ir_->subnet_ops.insert(op);

      if (runmode == RunMode::TPU_STATIC && !isa<ReturnOp>(op)) {
        auto end = module::getName(op);
        for (auto v : op->getOperands()) {
          if (v.getType().isa<NoneType>()) {
            continue;
          }
          if (v.isa<BlockArgument>()) {
            continue;
          }
          if (v.getDefiningOp() && isa<top::WeightOp>(v.getDefiningOp())) {
            continue;
          }
          auto start = module::getName(v);
          if (start == "image"){
            llvm::errs();
          }
          edges.push_back(std::make_pair(start.str(), end.str()));
        }
      }

      for (auto v : op->getOperands()) {
        if (v.getType().isa<NoneType>()) {
          continue;
        }
        lg_pass_ir_->subnet_values.insert(v);
      }
      for (auto v : op->getResults()) {
        bool is_used = false;
        for (auto dst_op : v.getUsers()) {
          if (lg_pass_ir_->subnet_ops.contains(dst_op)) {
            is_used = true;
            break;
          }
        }
        if (!is_used) {
          lg_pass_ir_->subnet_values.insert(v);
        }
      }
    }
  });


  // ==== do topo sort to build better ordered op list ====
  if (runmode == RunMode::TPU_STATIC){
    TopoSorter sorter;
    // 1. sort
    auto top_order = sorter.topologicalSortWithPriority(edges);

    // 2. validation and detection
    bool doReorder = true;
    if (lg_pass_ir_->subnet_ops.size() != (top_order.size() + 1)) {
      doReorder = false;
    } else {

      int oriCost = 0;
      int time = 0;
      std::unordered_map<std::string, int> oriOrder;
      for (auto it : llvm::enumerate(lg_pass_ir_->subnet_ops)) {
        if(!isa<ReturnOp>(it.value())){
          oriOrder[module::getName(it.value()).str()] = it.index();
        }
      }
      for (auto it : llvm::enumerate(lg_pass_ir_->subnet_ops)) {
        if(!isa<ReturnOp>(it.value())){
          oriCost += it.index() -
                    oriOrder[sorter.getParent(module::getName(it.value()).str())];
          time++;
        }
      }

      if (oriCost <= sorter.getCost() || time != sorter.getTime()){
        doReorder = false;
      }
    }

    // 3. do it
    if (doReorder) {
      // adjust lg_pass_ir_->subnet_ops
      std::vector<Operation *> vector(lg_pass_ir_->subnet_ops.size());
      for (auto op : lg_pass_ir_->subnet_ops) {
        if (!isa<ReturnOp>(op)) {
          vector[top_order[module::getName(op).str()]] = op;
        } else {
          vector[lg_pass_ir_->subnet_ops.size() - 1] = op;
        }
      }

      // adjust mlir context to avoid "does not dominate this use" problem
      lg_pass_ir_->subnet_ops.clear();
      for (auto it : llvm::enumerate(vector)) {
        auto op = it.value();
        lg_pass_ir_->subnet_ops.insert(op);
        if(it.index() >= 1){
          op->moveAfter(vector[it.index() - 1]);
          DEBUG_WITH_TYPE("topo_reorder_mlir", {
              llvm::dbgs() << "; action = topo"
              << "; before_op = " << module::getName(vector[it.index() - 1])
              << "; op = " << module::getName(op)
             << "\n";
          });
        } else {
            op->moveBefore(vector[it.index() + 1]);
            DEBUG_WITH_TYPE("topo_reorder_mlir", {
              llvm::dbgs() << "; action = topo"
                << "; before_op = " << module::getName(op)
                << "; op = " << module::getName(vector[it.index() + 1])
                << "\n";
            });
        }
        // op->dump();
        for(auto opd: op->getOperands()){
          auto opdOp = opd.getDefiningOp();
          if(opdOp && isa<top::WeightOp>(opdOp)){
            opdOp->moveBefore(op);
            DEBUG_WITH_TYPE("topo_reorder_mlir", {
              llvm::dbgs() << "; action = topo"
                << "; step = moveWeight"
                << "; weightOp = " << module::getName(opdOp)
                << "; op = " << module::getName(op)
                << "\n";
            });
          }
        }
      }

      auto &lastOp = func_.getBody().back().back();
      if(!isa<ReturnOp>(lastOp)){
        vector.back()->moveAfter(&lastOp);
      }
    }
  }

  if (opt != 3) {
    return;
  }

  lg_pass_ir_->branch_parallel = branch_parallel;
  if (branch_parallel) {
    int max_index = -1;
    Operation* last_op = nullptr;
    std::vector<std::pair<Value, Operation*>> out_var_def_ops;
    func.walk([&](Operation *op) {
      if (isa<tpu::YieldOp, ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto tmp_op = v.getDefiningOp();
          out_var_def_ops.push_back(std::make_pair(v, tmp_op));
          auto itr = std::find(lg_pass_ir_->subnet_ops.begin(), lg_pass_ir_->subnet_ops.end(), tmp_op);
          if (itr != lg_pass_ir_->subnet_ops.end()) {
              int index = std::distance(lg_pass_ir_->subnet_ops.begin(), itr);
              if (index > max_index) {
                max_index = index;
                last_op = tmp_op; //找到拓扑序的最后一个op
              }
          }
        }
      }
    });
    llvm::outs()<<"last_op name:"<<module::getName(last_op).str()<<"\n";
    lg_pass_ir_->subnet_return_opds.push_back(last_op->getResult(0));

    std::vector<Operation*> last_op_all_pre_ops;
    last_op_all_pre_ops.push_back(last_op);
    std::list<Value> tensor_branchs;
    for (auto v : last_op->getOperands()) {
      auto pre_op = v.getDefiningOp();
      if (pre_op != nullptr && isa<top::NoneOp, top::WeightOp, top::InputOp>(pre_op)) {
        continue;
      }
      last_op_all_pre_ops.push_back(pre_op);
      tensor_branchs.push_back(v);
    }

    while (!tensor_branchs.empty()) {
      auto out_tensor = tensor_branchs.front();
      tensor_branchs.pop_front();
      backward_collect_op(out_tensor, tensor_branchs, last_op_all_pre_ops);
    }

    for (auto itr: out_var_def_ops) {
      auto itr2 = std::find(last_op_all_pre_ops.begin(), last_op_all_pre_ops.end(), itr.second);
      if (itr2 == last_op_all_pre_ops.end()) {
        lg_pass_ir_->subnet_return_opds.push_back(itr.first);
      }
    }

    int i = 0;
    for (auto v : lg_pass_ir_->subnet_return_opds) {
      llvm::outs()<<"subnet_return_opds:"<<i++<<" name:"<<module::getName(v).str()<<"\n";
    }
  }

  std::vector<Operation*> global_layers, tmp_ops;
  for (auto itr: lg_pass_ir_->subnet_ops) {
    tmp_ops.push_back(itr);
  }
  func.walk([&](Operation *op) {
    if (isa<ReturnOp>(op)) {
      lg_pass_ir_->returnOp = op;
    } else if (!isa<FuncOp, top::NoneOp, top::WeightOp, top::InputOp>(op) && !isLgSupport(op)) {
      if (op->getResults().size() == 1 && is_value_weight(op->getResults()[0])) {
        tmp_ops.erase(std::remove(tmp_ops.begin(), tmp_ops.end(), op), tmp_ops.end());
        return WalkResult::skip();
      }
      global_layers.push_back(op);
      llvm::errs()<<"find global_layer: "<<show_op_info(op)<<"\n";
    }
    return WalkResult::advance();
  });

  GetAllParallelNodes(tmp_ops, lg_pass_ir_->map_parallel_op_subnet);
  for (auto grp: seg_grp_ops_by_global_op(tmp_ops, global_layers)) {
    lg_pass_ir_->tmp_base_groups.push_back(CreateIlpLgInfo(grp));
  }
}

void GroupOps::process(int64_t opt) {
  buildGroups(opt);
  if (opt == 3) {
    buildMlir_for_opt3();
  } else {
    buildMlir();
    if (module::isBM1688() &&
      (LgPass::OPTIONS.nnvlc_mode == NnvlcMode::ACTIVATION ||
       LgPass::OPTIONS.nnvlc_mode == NnvlcMode::ALL)) {
      buildNnvlcActivation();
    }
  }
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
  func_.walk([&](tpu::Conv2DOp op) {
    int use_3ic = op.getUse_3icOptimize();
    Operation *pre_op = op.getInput().getDefiningOp();
    if (use_3ic > 0 && pre_op && !isa<tpu::LoadOp>(pre_op)) {
      // broadcast input using BDC rather than GDMA
      if(!module::isBM1684Family()) use_3ic |= 0x10;
      op.setUse_3icOptimize(use_3ic);
    }
  });

  DEBUG_WITH_TYPE("dominate_bug", {
    module::getModuleOp().dump();
  });
}


typedef struct node_info {
  Operation* global_op = nullptr;
  LgInfo* lgInfo = nullptr;
  int indeg = 0;
  int idx = 0;
  std::vector<node_info*> pre_nodes;
  std::vector<node_info*> next_nodes;
  std::vector<node_info*> tmp_pre_nodes;
  node_info(Operation* global_op_):global_op(global_op_) {}
  node_info(LgInfo* lgInfo_):lgInfo(lgInfo_) {}
  bool operator==(const node_info &rhs) const {
    return global_op == rhs.global_op && lgInfo == rhs.lgInfo;
  }

  void show_info(std::string extra_info = "I am") {
    if (global_op) {
      LOG(INFO)<<extra_info <<" at global_op: "<<module::getName(global_op).str();
    } else {
      std::string tmp_str = "";
      int i = 0;
      for (auto it: lgInfo->group_ops) {
        if (i++ > 5) {
          break;
        }
        tmp_str = tmp_str + "---" + module::getName(it).str();
      }
      assert (tmp_str.size() > 0);
      LOG(INFO)<<extra_info  <<" at group:"<<tmp_str.substr(3, tmp_str.size());
    }
  }
} node_info;

bool node_info_Sort_by_int(const node_info &v1, const node_info &v2)
{
    return v1.idx < v2.idx;//升序排列
}

static void nodes_topo_order_dfs(node_info& cur_node, std::vector<node_info>& topo_nodes, int& idx) {
  assert (cur_node.global_op || cur_node.lgInfo);
  cur_node.idx = idx++;
  topo_nodes.push_back(cur_node);
  // cur_node.show_info("add into topo_nodes, idx:"+std::to_string(idx - 1));
  for (auto next_node : cur_node.next_nodes) {
    next_node->indeg -=1;
    auto& nodes = next_node->tmp_pre_nodes;
    nodes.erase(std::remove(nodes.begin(), nodes.end(), &cur_node), nodes.end());
    // next_node->show_info("check next_node, indeg:"+std::to_string(next_node->indeg));
    if (next_node->indeg == 0) {
      if (std::find(topo_nodes.begin(), topo_nodes.end(), *next_node) == topo_nodes.end()) {
        // LOG(INFO) <<"  call nodes_topo_order_dfs";
        nodes_topo_order_dfs(*next_node, topo_nodes, idx);
      }
    }
  }
}

void GroupOps::buildMlir_for_opt3() {
  auto &lg_infos = lg_pass_ir_->lg_infos;
  if (lg_infos.empty()) {
    return;
  }

  groups_.clear();
  int64_t group_num = lg_infos.size();
  auto builder = OpBuilder(ctx_);

  //添加group node
  std::vector<node_info> nodes;
  std::vector<Operation*> all_local_ops;
  std::map<node_info*, std::vector<node_info*>> map_parallel_node_subnet;
  for (int64_t i = 0; i < group_num; i++) {
    if (lg_infos[i].group_ops.size() > 1) {
      node_info tmp(&lg_infos[i]);
      tmp.show_info("add group");
      nodes.push_back(tmp);
      all_local_ops.insert(all_local_ops.begin(), lg_infos[i].group_ops.begin(), lg_infos[i].group_ops.end());

      for (auto op : lg_infos[i].group_ops) {
        llvm::outs() << "op: "<<module::getName(op).str()<<"\n";
      }
      for (auto out: lg_infos[i].group_outs) {
        llvm::outs() <<"   group_outs: "<<module::getName(out).str()<<"\n";
      }
      for (auto in: lg_infos[i].group_ins) {
        llvm::outs() <<"   group_ins: "<<module::getName(in).str()<<"\n";
      }
    }
  }

  //添加global node
  func_.walk([&](Operation *op) {
    if (!isa<FuncOp, ReturnOp, top::NoneOp>(op)) {
      if (std::find(all_local_ops.begin(), all_local_ops.end(), op) == all_local_ops.end()) {
        LOG(INFO) <<"add global_op: "<<module::getName(op).str();
        nodes.push_back(node_info(op));
      }
    }
  });
  Operation* retrunOp = nullptr;
  func_.walk([&](Operation *op) {
    if (isa<ReturnOp>(op)) {
      retrunOp = op;
    }
  });

  //获取node的前后驱node
  LOG(INFO) <<"get pre&next nodes, nodes.size: "<<nodes.size();
  for (auto& node: nodes) {
    if (node.global_op == nullptr) {
      node.show_info();
      for (auto in: node.lgInfo->group_ins) {
        auto in_op = in.getDefiningOp();
        if (in_op && !isa<top::NoneOp>(in_op)) {
          for (auto& node2: nodes) {
            if (node == node2)
              continue;
            if (node2.global_op) {
              if (in_op == node2.global_op) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(), &node2) == node.pre_nodes.end()) {
                  LOG(INFO) <<"find pre node: "<<module::getName(in_op).str();
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            } else {
              auto& node_ops = node2.lgInfo->group_ops;
              if (std::find(node_ops.begin(), node_ops.end(), in_op) != node_ops.end()) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(), &node2) == node.pre_nodes.end()) {
                  LOG(INFO) <<"find pre group: "<<module::getName(in_op).str();
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            }
          }
        }
      }
      for (auto out: node.lgInfo->group_outs) {
        for (auto user: out.getUsers()) {
          if (isa<ReturnOp>(user)){
            continue;
          }
          for (auto& node2: nodes) {
            if (node == node2)
              continue;
            if (node2.global_op) {
              if (user == node2.global_op) {
                if (std::find(node.next_nodes.begin(), node.next_nodes.end(), &node2) == node.next_nodes.end()) {
                  LOG(INFO) <<"find next node: "<<module::getName(user).str();
                  node.next_nodes.push_back(&node2);
                  break;
                }
              }
            } else {
              auto& node_ops = node2.lgInfo->group_ops;
              if (std::find(node_ops.begin(), node_ops.end(), user) != node_ops.end()) {
                if (std::find(node.next_nodes.begin(), node.next_nodes.end(), &node2) == node.next_nodes.end()) {
                  LOG(INFO) <<"find next group, have:"<<module::getName(user).str();
                  node.next_nodes.push_back(&node2);
                  break;
                }
              }
            }
          }
        }
      }
    } else {
      // LOG(INFO) <<"check node: "<<module::getName(node.global_op).str();
      node.show_info();
      for (auto in: node.global_op->getOperands()) {
        auto in_op = in.getDefiningOp();
        if (in_op && !isa<top::NoneOp>(in_op)) {
          for (auto& node2: nodes) {
            if (node == node2)
              continue;
            if (node2.global_op) {
              if (in_op == node2.global_op) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(), &node2) == node.pre_nodes.end()) {
                  LOG(INFO) <<"find pre node: "<<module::getName(in_op).str();
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            } else {
              auto& node_ops = node2.lgInfo->group_ops;
              if (std::find(node_ops.begin(), node_ops.end(), in_op) != node_ops.end()) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(), &node2) == node.pre_nodes.end()) {
                  LOG(INFO) <<"find pre group: "<<module::getName(in_op).str();
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            }
          }
        }
      }
      for (auto user: node.global_op->getUsers()) {
        if (isa<ReturnOp>(user)){
          continue;
        }
        for (auto& node2: nodes) {
          if (node == node2)
            continue;
          if (node2.global_op) {
            if (user == node2.global_op) {
              if (std::find(node.next_nodes.begin(), node.next_nodes.end(), &node2) == node.next_nodes.end()) {
                LOG(INFO) <<"find next node: "<<module::getName(user).str();
                node.next_nodes.push_back(&node2);
                break;
              }
            }
          } else {
            auto& node_ops = node2.lgInfo->group_ops;
            if (std::find(node_ops.begin(), node_ops.end(), user) != node_ops.end()) {
              if (std::find(node.next_nodes.begin(), node.next_nodes.end(), &node2) == node.next_nodes.end()) {
                LOG(INFO) <<"find next group, have:"<<module::getName(user).str();
                node.next_nodes.push_back(&node2);
                break;
              }
            }
          }
        }
      }
    }
  }
  LOG(INFO) <<"nodes.size: "<<nodes.size();
  //设置node的入度
  for (auto& node : nodes) {
    node.indeg = node.pre_nodes.size();
    node.tmp_pre_nodes.assign(node.pre_nodes.begin(), node.pre_nodes.end());
  }
  //深度搜索node的topo序
  int idx = 0;
  std::vector<node_info> topo_nodes;
  for (auto node : nodes) {
    if (node.indeg == 0) {
      if (std::find(topo_nodes.begin(), topo_nodes.end(), node) == topo_nodes.end()) {
        // node.show_info("dfs start point");
        nodes_topo_order_dfs(node, topo_nodes, idx);
      }
    }
  }

  //lg_pass_ir_->map_parallel_node_subnet

  std::sort(topo_nodes.begin(), topo_nodes.end(), node_info_Sort_by_int);
  LOG(INFO) <<"topo_nodes.size: "<<topo_nodes.size();
  for (auto& node : nodes) {
    if (node.tmp_pre_nodes.size() > 0) {
      node.show_info("have untrack pre_nodes:");
      for (auto node2 : node.tmp_pre_nodes) {
        node2->show_info("  ");
      }
    }
  }

  typedef struct lg_extra_info {
    std::vector<Operation*> slice_merge_ops;
    std::vector<Operation*> grp_group_ops;
    std::vector<Operation*> outbuffer_ops;
  } lg_extra_info;

  std::vector<std::string> need_dump_tensors = {"not_dump"};
  if (module::isDebugCmdEnable("dump_group_all_outs_for_cmp")) {
    need_dump_tensors = {"dump_all"};
  }

  std::map<int64_t, lg_extra_info> map_lg_extra_info;
  for (int64_t i = 0; i < group_num; i++) {
    if (lg_infos[i].group_ops.size() > 1) {
      lg_extra_info tmp;
      map_lg_extra_info[(int64_t)&lg_infos[i]] = tmp;
    }
  }

  map_old_grp_out_to_new_grp_out.clear();
  std::map<Value, Value, value_compare> map_new_grp_out_to_old_grp_out;
  for (int64_t group_idx = 0; group_idx < group_num; group_idx++) {
    if (lg_infos[group_idx].group_ops.size() > 1 && lg_pass_ir_->ILP_time_steps[group_idx].size() > 0) {
      map_name_output_to_merge_slice_for_grp.clear();
      std::map<Value, std::vector<Value>, value_compare> map_output_to_merge_slice_for_grp;
      map_store_tensor_to_outbuffer_out.clear();
      const LgInfo &lg_info = lg_infos[group_idx];
      TensorInfo &tensor_info = lg_pass_ir_->lg_tensor_infos_[group_idx];
      auto &ops = lg_info.group_ops;
      auto &op_outs = lg_info.group_op_outs;
      current_op_ = ops.back();
      need_store_load_value.clear();
      will_store_value.clear();
      std::vector<Value> need_dump_tensor_values;
      for (int pipe_id = 0; pipe_id < lg_pass_ir_->ILP_time_steps[group_idx].size(); pipe_id++) {
        LOG(INFO) <<"preprocess grp: " << group_idx << ", unique pipe_id:"<<pipe_id;
        ILP_time_step = lg_pass_ir_->ILP_time_steps[group_idx][pipe_id];
        int ts_count = ILP_time_step->ts_count;
        std::vector<Value> tmp_need_store_load_value;
        for (size_t ts = 0; ts < ts_count; ++ts) {
          LOG(INFO) <<"----------------ts"<<ts<<"-----------------";
          for (auto it: ILP_time_step->timestep_table_new[ts].vec_ts_var) {
            if (it.var_value == 1) {
              auto itr = ILP_time_step->values_need_store_to_grpout.begin();
              for (;itr != ILP_time_step->values_need_store_to_grpout.end(); ++itr) {
                // if (it.varName == itr->second) { //new edit
                if (std::find(itr->second.begin(), itr->second.end(), it.varName) != itr->second.end()) {
                  tmp_need_store_load_value.push_back(itr->first);
                  if (map_store_tensor_to_outbuffer_out.find(itr->first) == map_store_tensor_to_outbuffer_out.end()) {
                    assert(itr->first == it.value);
                    builder.setInsertionPointAfter(current_op_);
                    auto loc = NameLoc::get(builder.getStringAttr(module::getName(itr->first).str()));
                    auto out_grp_op = builder.create<tpu::OutBufferOp>(loc, itr->first.getType(), ValueRange{});
                    map_lg_extra_info[(int64_t)&lg_info].outbuffer_ops.push_back(out_grp_op);
                    map_store_tensor_to_outbuffer_out[itr->first] = out_grp_op.getResult();
                    current_op_ = out_grp_op;
                    // LOG(INFO) <<"out_grp_op.dump:";
                    // out_grp_op.dump();
                  }
                  break;
                }
              }

              if (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD) {
                tmp_need_store_load_value.push_back(it.value);
              }
            }
          }
        }

        std::vector<Value> tmp_will_store_value;
        for (size_t ts = 0; ts < ts_count; ts++) {
          LOG(INFO) <<"----------------collect will_store_value at ts"<<ts<<"-----------------";
          for (auto it: ILP_time_step->timestep_table_new[ts].vec_ts_var) {
            if (it.var_value == 1) {
              auto out_name = module::getName(it.value).str();
              if (it.info.mode2 & TIMESTEP2_STORE) {
                LOG(INFO) <<"add into will_store_value for store, name:"<<out_name;
                tmp_will_store_value.push_back(it.value);
              } else if (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD
                        && ILP_time_step->mapILPVarInfo[it.varName].store_load_mode == 0) {
                LOG(INFO) <<"add into will_store_value for store_and_load, name:"<<out_name;
                tmp_will_store_value.push_back(it.value);
              }
            }
          }
        }
        will_store_value.push_back(tmp_will_store_value);

        for (auto op_out: op_outs) {
          if (need_dump_tensors[0] == "not_dump" ||
            std::find(ops.begin(), ops.end(), op_out.getDefiningOp()) == ops.end()) {
            continue;
          }
          auto out_name = module::getName(op_out).str();
          bool not_need_create = std::find(tmp_will_store_value.begin(), tmp_will_store_value.end(), op_out)
                                          != tmp_will_store_value.end();
          if (need_dump_tensors[0] == "dump_all" ||
            std::find(need_dump_tensors.begin(), need_dump_tensors.end(), out_name) != need_dump_tensors.end()) {
            if (!not_need_create) {
              tmp_need_store_load_value.push_back(op_out);
            }
            if (map_store_tensor_to_outbuffer_out.find(op_out) == map_store_tensor_to_outbuffer_out.end()) {
              if (!not_need_create) {
                need_dump_tensor_values.push_back(op_out);
                builder.setInsertionPointAfter(current_op_);
                auto loc = NameLoc::get(builder.getStringAttr(out_name));
                auto out_grp_op = builder.create<tpu::OutBufferOp>(loc, op_out.getType(), builder.getBoolAttr(true));
                map_lg_extra_info[(int64_t)&lg_info].outbuffer_ops.push_back(out_grp_op);
                map_store_tensor_to_outbuffer_out[op_out] = out_grp_op.getResult();
                current_op_ = out_grp_op;
                // LOG(INFO) <<"out_grp_op.dump2:";
                // out_grp_op.dump();
              }
            } else {
              auto whOp = map_store_tensor_to_outbuffer_out[op_out].getDefiningOp();
              llvm::outs() <<"change need_dump for " << module::getName(op_out)<<"\n";
              whOp->setAttr("need_dump", builder.getBoolAttr(true));
            }
          }
        }
        llvm::outs() <<"tmp_need_store_load_value.size:" << tmp_need_store_load_value.size()<<"\n";
        need_store_load_value.push_back(tmp_need_store_load_value);
      }

      for (auto out : lg_info.group_outs) {
        map_output_to_merge_slice_for_grp[out] = std::vector<Value>();
        map_name_output_to_merge_slice_for_grp[out] = std::vector<std::string>();
      }
      for (auto out : lg_info.group_op_outs) {
        map_old_to_new_value[out] = std::map<int, Value>();
      }
      bool one_grp = lg_pass_ir_->ILP_time_steps[group_idx].size() == 1;

      llvm::outs() << "group"<<group_idx<< ", group_id:"<<lg_info.group_id<<", in and out:\n";
      for (auto op : lg_info.group_ops) {
        llvm::outs() << "  op:"<<show_op_info(op)<<"\n";
      }
      for (auto out: lg_info.group_outs) {
        llvm::outs() << "    out:"<<module::getName(out).str()<<"\n";
      }
      for (auto in: lg_info.group_ins) {
        llvm::outs() << "    in:"<<module::getName(in).str()<<"\n";
      }
      assert(lg_pass_ir_->ILP_time_steps[group_idx].size() > 0);
      auto& l2mem_alloc_ptr = lg_pass_ir_->lg_l2mem_alloc_ptr[group_idx];
      for (int pipe_id = 0; pipe_id < lg_pass_ir_->ILP_time_steps[group_idx].size(); pipe_id++) {
        LOG(INFO) <<"grp: " << group_idx << ", unique pipe_id:"<<pipe_id;
        ILP_time_step = lg_pass_ir_->ILP_time_steps[group_idx][pipe_id];
        const shape_secs_t &shape_secs = lg_pass_ir_->shape_secs[group_idx];

        llvm::SmallVector<Value, 80> operands;
        llvm::SmallVector<Value, 80> outputs;
        llvm::SmallVector<NamedAttribute, 80> attrs;
        llvm::SmallVector<Type, 80> in_types;
        llvm::SmallVector<Type, 80> ret_types;
        int64_t nsecs = shape_secs.nsecs;
        int64_t hsecs = shape_secs.hsecs;
        int64_t dsecs = shape_secs.dsecs;
        int64_t wsecs = shape_secs.wsecs;
        int64_t csecs = shape_secs.csecs;
        for (auto in : lg_info.group_ins) {
          in_types.push_back(in.getType());
          if (map_old_grp_out_to_new_grp_out.find(in) != map_old_grp_out_to_new_grp_out.end()) {
            operands.push_back(map_old_grp_out_to_new_grp_out[in]);
          } else {
            operands.push_back(in);
          }
        }
        for (auto in : need_store_load_value[pipe_id]) {
          if (map_store_tensor_to_outbuffer_out.find(in) != map_store_tensor_to_outbuffer_out.end()) {
            auto outbuffer_out = map_store_tensor_to_outbuffer_out[in];
            in_types.push_back(outbuffer_out.getType());
            operands.push_back(outbuffer_out);
          }
        }
        std::map<Value, std::vector<Value>, value_compare> map_output_to_merge_slice;
        std::map<Value, Value, value_compare> map_group_out_to_yield_in;
        llvm::SmallVector<NameLoc, 80> nameLocs;
        llvm::SmallVector<Location, 80> locs;
        for (auto out : lg_info.group_outs) {
          ret_types.push_back(out.getType());
          auto tmpStr = module::getLoc(out).getName().str();
          if (!one_grp) {
            tmpStr = tmpStr + "_pipe" + std::to_string(pipe_id);
          }
          LOG(INFO) <<" new group_out name:"<<tmpStr;
          auto newName = NameLoc::get(builder.getStringAttr(tmpStr));
          locs.push_back(newName);
          nameLocs.push_back(newName);
          outputs.push_back(out);
          map_output_to_merge_slice[out] = std::vector<Value>();
        }
        std::vector<int64_t> core_ids;
        std::vector<int64_t> core_slice_ncdhws;
        for (const auto& pair : ILP_time_step->ncdhw_steps) {
          core_ids.push_back(pair.first);
          for (auto per_slice : pair.second) {
            for (auto per_int : per_slice) {
                core_slice_ncdhws.push_back(per_int);
            }
            core_slice_ncdhws.push_back(-1);
          }
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
        attrs.push_back(
            builder.getNamedAttr("run_core_id", builder.getI64ArrayAttr(core_ids)));
        attrs.push_back(
            builder.getNamedAttr("core_slice_ncdhw", builder.getI64ArrayAttr(core_slice_ncdhws)));
        attrs.push_back(builder.getNamedAttr(
            "group_type", builder.getI64IntegerAttr((int64_t)lg_info.type)));

        builder.setInsertionPointAfter(current_op_);
        auto groupOp =
            builder.create<tpu::GroupOp>(group_loc, ret_types, operands, attrs);
        map_lg_extra_info[(int64_t)&lg_info].grp_group_ops.push_back(groupOp);
        // LOG(INFO) <<"groupOp: ";
        // groupOp->dump();
        body_ = new Block();
        groupOp.getBody().push_back(body_);

        //  replace outputs
        for (auto it : llvm::enumerate(groupOp.getResults())) {
          if (one_grp) {
            outputs[it.index()].replaceUsesWithIf(it.value(), [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              return find(ops.begin(), ops.end(), user) == ops.end();
            }); //旧的grp outputs用新的groupOp results替换，这些grp output转为OpOperand，其拥有者不在grp中，即组外user的opd会替换
            map_old_grp_out_to_new_grp_out[outputs[it.index()]] = it.value();
            map_new_grp_out_to_old_grp_out[it.value()] = outputs[it.index()];
          } else {
            map_output_to_merge_slice_for_grp[outputs[it.index()]].push_back(it.value());
          }
        }

        current_op_ = nullptr;
        map_store_to_load_value.clear();
        map_l2m_out_to_load_in.clear();
        int ts_count = ILP_time_step->ts_count;
        bool one_slice = ILP_time_step->slice_num == 1;

        for (size_t ts = 0, ts_id = 0; ts < ts_count; ts++, ts_id++) {
          if (lg_pass_ir_->map_l2m_loads.size() > 0) {
            auto& map_l2m_load = lg_pass_ir_->map_l2m_loads[group_idx];
            if (ts == 0) {
              if (map_l2m_load.find(-1) != map_l2m_load.end()) {
                LOG(INFO) <<"----------------ts"<<ts_id<<" for CreateLoadToL2mOp-----------------";
                for (auto it: map_l2m_load[-1]) {
                  CreateLoadToL2mOp(ts_id, it, pipe_id, l2mem_alloc_ptr);
                }
                ts_id++;
              }
            }
            if (map_l2m_load.find(ts) != map_l2m_load.end()) {
              LOG(INFO) <<"----------------ts"<<ts_id<<" for CreateLoadToL2mOp-----------------";
              for (auto it: map_l2m_load[ts]) {
                CreateLoadToL2mOp(ts_id, it, pipe_id, l2mem_alloc_ptr);
              }
            }
          }
          bool can_merge = ILP_time_step->timestep_table_new[ts].can_merge;
          if (ILP_time_step->inserted_timestep_table_.find(ts) != ILP_time_step->inserted_timestep_table_.end()) {
            LOG(INFO) <<"----------------ts"<<ts_id<<" for CreateLmemMoveOp-----------------";
            CreateLmemMoveOp(ts_id, ILP_time_step->inserted_timestep_table_[ts][0]);
            ts_id++;
          }
          LOG(INFO) <<"----------------ts"<<ts_id<<"-----------------";
          for (auto it: ILP_time_step->timestep_table_new[ts].vec_ts_var) {
            if (it.var_value == 1) {
              std::vector<int64_t> ncdhw_idx = ILP_time_step->ncdhw_steps.begin()->second[it.slice_idx];
              if (it.info.mode2 & TIMESTEP2_LOAD) {
                CreateLoadOp2(ts_id, it, pipe_id, ops, ncdhw_idx, lg_info, can_merge);
              } else if (it.info.mode2 & TIMESTEP2_STORE) {
                auto storeOp_out = CreateStoreOp2(it.value, it.info, ts_id, it.slice_idx, pipe_id, lg_info.type, can_merge);
                map_output_to_merge_slice[it.value].push_back(storeOp_out);
                map_group_out_to_yield_in[it.value] = storeOp_out;
              } else if (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD) {
                if (ILP_time_step->mapILPVarInfo[it.varName].store_load_mode == 0) {
                  auto storeOp_out = CreateStoreOp2(it.value, it.info, ts_id, it.slice_idx, pipe_id, lg_info.type, can_merge);
                  if (std::find(lg_info.group_outs.begin(), lg_info.group_outs.end(), it.value)
                            != lg_info.group_outs.end()) {
                    map_output_to_merge_slice[it.value].push_back(storeOp_out);
                    map_group_out_to_yield_in[it.value] = storeOp_out;
                  }
                } else {
                  CreateLoadOp2(ts_id, it, pipe_id, ops, ncdhw_idx, lg_info, can_merge);
                }
              }
            }
          }

          for (auto it: ILP_time_step->timestep_table_new[ts].vec_op_infos) {
            builder.setInsertionPointAfter(current_op_);
            auto new_op = builder.clone(*it.op);
            std::string tmpStr = "pipe" +std::to_string(pipe_id)  + "_slice" + std::to_string(it.slice_idx);
            module::setLocSuffix(new_op, tmpStr);
            std::string name = module::getName(new_op).str();
            LOG(INFO) <<"bdc_op_name: " << name;
            for (OpOperand &opd : it.op->getOpOperands()) {
              auto pre_op = opd.get().getDefiningOp();
              if (pre_op != nullptr && isa<top::NoneOp>(pre_op)) {
                continue;
              }
              Value new_value = opd.get();
              if (map_new_grp_out_to_old_grp_out.find(opd.get()) != map_new_grp_out_to_old_grp_out.end()) {
                //opd已被替换为前面group的输出
                new_value = map_old_to_new_value[map_new_grp_out_to_old_grp_out[opd.get()]][it.slice_idx];
              } else {
                if (map_old_to_new_value.find(opd.get()) != map_old_to_new_value.end()) {
                  auto tmp_map = map_old_to_new_value[opd.get()];
                  if (tmp_map.find(it.slice_idx) != tmp_map.end()) {
                    new_value = tmp_map[it.slice_idx];
                  } else {
                    new_value = tmp_map[0]; //使用驻留的权重
                  }
                }
              }
              new_op->setOperand(opd.getOperandNumber(), new_value);
            }
            for (const auto &res : llvm::enumerate(it.op->getResults())) {
              map_old_to_new_value[res.value()][it.slice_idx] = new_op->getResult(res.index());
            }
            auto ncdhw_idx = ILP_time_step->ncdhw_steps.begin()->second[it.slice_idx];
            UpdateOpLgParam2(new_op, it.op, ts_id, it.slice_idx, tensor_info, ncdhw_idx, lg_info.type, can_merge);
            current_op_ = new_op;

            bool first_time = true;
            //针对复合op的情形，下面处理应该在完成vec_op_infos的遍历后再统一处理，目前的处理会在复合op中插入多个dump的时隙
            //但考虑到dump只是调试功能，不会影响正常运行的性能，而目前的实现大致是可行的，也还没有发现bug，故暂不修改
            for (auto res : llvm::enumerate(it.op->getResults())) {
              if (std::find(will_store_value[pipe_id].begin(), will_store_value[pipe_id].end(), res.value())
                            == will_store_value[pipe_id].end()) {
                if (std::find(need_dump_tensor_values.begin(), need_dump_tensor_values.end(), res.value())
                              != need_dump_tensor_values.end()) {
                  if (first_time) {
                    ts_id++;
                    LOG(INFO) <<"----------------ts"<<ts_id<<" for op result dump-----------------";
                    first_time = false;
                  }
                  CreateStoreOp2(res.value(), tensor_info[res.value()], ts_id, it.slice_idx, pipe_id, lg_info.type, false);
                }
              }
            }
          }
        }
        if (!one_slice) {
          LOG(INFO) <<"groupOp have many slice";
          map_group_out_to_yield_in.clear();
          builder.setInsertionPointAfter(current_op_);
          attrs.clear();
          int m = 0;
          for (auto out : lg_info.group_outs) {
            auto out_slice_add_op = builder.create<tpu::SliceMergeOp>(
                locs[m], out.getType(), map_output_to_merge_slice[out], attrs);
            map_group_out_to_yield_in[out] = out_slice_add_op.getOutput();
            current_op_ = out_slice_add_op;
            m++;
          }
        }
        llvm::SmallVector<Value, 18> new_stores;
        for (auto out : lg_info.group_outs) {
          new_stores.push_back(map_group_out_to_yield_in[out]);
        }
        builder.setInsertionPointAfter(current_op_);
#if 0
        for (auto [i, new_store] : llvm::enumerate(new_stores)) {
          LOG(INFO) <<"idx:"<<i<<", store value:";
          new_store.dump();
        }
        LOG(INFO) <<"group_loc:";
        group_loc.dump();
#endif
        builder.create<tpu::YieldOp>(group_loc, new_stores);
        LOG(INFO) <<"groupOp.dump: ";
        groupOp.dump();
        current_op_ = groupOp; //下一个group插入在当前group后
      }
      if (!one_grp) {
        builder.setInsertionPointAfter(current_op_);
        llvm::SmallVector<NamedAttribute, 18> attrs;
        for (auto out : lg_info.group_outs) {
          auto tmp_op = builder.create<tpu::SliceMergeOp>(
              module::getLoc(out), out.getType(), map_output_to_merge_slice_for_grp[out], attrs);
          out.replaceUsesWithIf(tmp_op.getOutput(), [&](OpOperand &operand) {
            Operation *user = operand.getOwner();
            return find(ops.begin(), ops.end(), user) == ops.end();
          });
          map_old_grp_out_to_new_grp_out[out] = tmp_op.getOutput();
          map_new_grp_out_to_old_grp_out[tmp_op.getOutput()] = out;
          map_lg_extra_info[(int64_t)&lg_info].slice_merge_ops.push_back(tmp_op);
        }
      }
    } else {
      // assert(false);
    }
  }

  if (module::isDebugCmdEnable("print_node_topo_change")) {
    idx = 0;
    for (auto node: topo_nodes) {
      if (node.global_op) {
        LOG(INFO) <<"idx: "<<idx<<" global_op: "<<module::getName(node.global_op).str();
      } else {
        LOG(INFO) <<"idx: "<<idx<<" group:";
        for (auto it: node.lgInfo->group_ops) {
          LOG(INFO) <<"  local op: "<<module::getName(it).str();
        }
      }
      idx++;
    }
  }
  Operation* firstOp = module::getNoneOp(lg_infos[0].group_ops[0]);
  for (auto node: topo_nodes) {
    if (node.global_op) {
      node.global_op->moveAfter(firstOp);
      if (module::isDebugCmdEnable("print_node_topo_change")) {
        if (isa<top::NoneOp>(firstOp))
          LOG(INFO) <<"  move node, from: "<<module::getName(node.global_op).str()<<", to:NoneOp";
        else
          LOG(INFO) <<"  move node, from: "<<module::getName(node.global_op).str()<<", to:"<<module::getName(firstOp).str();
      }
      firstOp  = node.global_op;
    } else {
      for (auto it2: map_lg_extra_info[(int64_t)node.lgInfo].outbuffer_ops) {
        it2->moveAfter(firstOp);
        if (module::isDebugCmdEnable("print_node_topo_change")) {
          if (isa<top::NoneOp>(firstOp))
            LOG(INFO) <<"  move group, from: "<<module::getName(it2).str()<<", to:NoneOp";
          else
            LOG(INFO) <<"  move group, from: "<<module::getName(it2).str()<<", to:"<<module::getName(firstOp).str();
        }
        firstOp  = it2;
      }
      for (auto it2: map_lg_extra_info[(int64_t)node.lgInfo].grp_group_ops) {
        it2->moveAfter(firstOp);
        if (module::isDebugCmdEnable("print_node_topo_change")) {
          if (isa<top::NoneOp>(firstOp))
            LOG(INFO) <<"  move group, from: "<<module::getName(it2).str()<<", to:NoneOp";
          else
            LOG(INFO) <<"  move group, from: "<<module::getName(it2).str()<<", to:"<<module::getName(firstOp).str();
        }
        firstOp  = it2;
      }
      for (auto it2: map_lg_extra_info[(int64_t)node.lgInfo].slice_merge_ops) {
        it2->moveAfter(firstOp);
        firstOp  = it2;
      }
    }
  }
  retrunOp->moveBefore(&retrunOp->getBlock()->back());

  //设置node的入度
  /*for (auto& node : nodes) {
    node.indeg = node.pre_nodes.size();
  }
  //深度搜索node的topo序
  int idx = 0;
  std::vector<node_info> topo_nodes;
  while(true) {
    int core_num = 8;
    for (auto node : nodes) {
      if (node.indeg == 0) {
        if (!node.global_op) {
          int used_core_num = 8 - node.lgInfo->free_cores.size();
          if (core_num >= used_core_num) {
            core_num -= used_core_num;
          } else {
            break;
          }
        } else {
          if (node.global_op->isSupportedMultiCore()) {
            node.global_op->setCoreNum(core_num);
            break;
          } else {
            core_num--;
          }
        }
      }
    }
  }*/
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
          if (current_op_ !=nullptr) {
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

void GroupOps::CreateLmemMoveOp(int64_t ts, ts_move_info& move_info) {
  assert(current_op_ != nullptr);
  auto builder = OpBuilder(ctx_);
  builder.setInsertionPointAfter(current_op_);

  std::vector<Value> operands;
  int i = 0;
  for (auto itr: move_info.move_value) {
    LOG(INFO) <<"need move value:"<<module::getName(itr).str();
    auto new_value = map_old_to_new_value[itr][move_info.slice_idx[i]];
    if (map_l2m_out_to_load_in.find(itr) != map_l2m_out_to_load_in.end()){
      new_value = map_l2m_out_to_load_in[itr];
    }
    operands.push_back(new_value);
    i++;
  }

  llvm::SmallVector<Type, 20> ret_types;
  llvm::SmallVector<Location, 20> locs;
  for (auto out : operands) {
    ret_types.push_back(out.getType());
    auto tmpStr = module::getLoc(out).getName().str() + "_moved";
    auto newName = NameLoc::get(builder.getStringAttr(tmpStr));
    locs.push_back(newName);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr(
      "move_src_add", builder.getI64ArrayAttr(move_info.move_src_add)));
  attrs.push_back(builder.getNamedAttr(
      "move_dest_add", builder.getI64ArrayAttr(move_info.move_dest_add)));
  attrs.push_back(builder.getNamedAttr(
      "move_size", builder.getI64ArrayAttr(move_info.move_size)));
  attrs.push_back(builder.getNamedAttr(
      "ts_id", builder.getI64IntegerAttr(ts)));
  attrs.push_back(builder.getNamedAttr(
      "name", builder.getStringAttr(move_info.name)));
  auto op_loc = builder.getFusedLoc(locs);
  auto moveOp = builder.create<tpu::MoveOp>(op_loc, ret_types, operands, attrs);

  i = 0;
  for (auto itr: move_info.move_value) {
    map_old_to_new_value[itr][move_info.slice_idx[i]] = moveOp->getResult(i);
    // LOG(INFO) <<"xxxxx12, name:"<<itr.name<<", value:"<<module::getName(itr.value).str()<<", slice:"<<itr.slice_idx;
    i++;
  }
  current_op_ = moveOp;
}


void GroupOps::CreateLoadToL2mOp(int64_t ts, l2m_value_info& it, int64_t pipe_id, l2mem_alloc_Ptr l2mem_alloc_ptr) {
  Value &input = it.value;
  std::string in_name = module::getName(input).str();
  int64_t slice_idx = it.slice_idx;
  if (!it.valid) {
    llvm::outs() <<"CreateLoadToL2mOp, name:"<<in_name<<", slice_idx:"<<slice_idx<<"invalid, skiped\n";
    return;
  }
  auto builder = OpBuilder(ctx_);
  std::vector<NamedAttribute> attrs;
  std::string name = "loadToL2m_";
  name = name + in_name;
  name = name + "_pipe" +std::to_string(pipe_id) + "_slice"+ std::to_string(slice_idx);

  std::string key = llvm::formatv("{0}_slice{1}", in_name, slice_idx).str();
  if (l2mem_alloc_ptr->vec_mem_alloc_his.find(key) == l2mem_alloc_ptr->vec_mem_alloc_his.end()) {
    key = llvm::formatv("{0}_slice-1", in_name).str();
  }
  auto l2mem_s_addr = l2mem_alloc_ptr->vec_mem_alloc_his[key].vec_reload_addr[0].second.addr;
  attrs.push_back(builder.getNamedAttr("l2m_addr", builder.getI64IntegerAttr(l2mem_s_addr)));
  attrs.push_back(
      builder.getNamedAttr("id", builder.getI64IntegerAttr(ts)));
  if (current_op_ == nullptr) {
    builder.setInsertionPointToStart(body_);
  } else {
    builder.setInsertionPointAfter(current_op_);
  }

  std::vector<Value> operands;
  operands.push_back(input);
  auto loadToL2mOp = builder.create<tpu::LoadToL2MOp>(NameLoc::get(builder.getStringAttr(name)),
                input.getType(), operands, attrs);
  map_l2m_out_to_load_in[input] = loadToL2mOp->getResult(0);
  current_op_ = loadToL2mOp;
}

// std::string my_to_string(int num, int print_width = 0) {
//   std::stringstream ss;
//   if (print_width == 0)
//     ss << num;
//   else
//     ss << std::setw(print_width) << std::setfill('0') << num;
//   return ss.str();
// }

void GroupOps::CreateLoadOp2(int64_t ts, ts_var_t& ts_var, int64_t pipe_id,
                            const std::vector<Operation *> &ops, std::vector<int64_t> ncdhw_idx,
                            const LgInfo& lgInfo, bool can_merge) {
  Value &input = ts_var.value;
  tensor_info_t& ti = ts_var.info;
  int64_t slice_idx = ts_var.slice_idx;
  auto builder = OpBuilder(ctx_);
  auto inputOp = input.getDefiningOp();
  std::vector<NamedAttribute> attrs;
  bool train = true;
  std::string name = "load_";
  std::string in_name = module::getName(input).str();
  LOG(INFO) <<"load_value: " << in_name;
  if (inputOp != nullptr) {
    name = name + in_name;
  } else {
    int arg_idx = input.cast<BlockArgument>().getArgNumber();
    name = name + std::to_string(arg_idx);
  }
  name = name + "_pipe" +std::to_string(pipe_id) + "_slice"+ std::to_string(slice_idx);
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
    if (weightop.getAllowSplitAttr() != nullptr){
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
    if (train) {
      buffer_key.type = LMEM_WEIGHT;
      attrs.push_back(builder.getNamedAttr(
          "lmem_type", builder.getI64IntegerAttr(LMEM_WEIGHT)));
    } else {
      attrs.push_back(builder.getNamedAttr(
          "lmem_type", builder.getI64IntegerAttr(LMEM_ACTIVATION)));
    }
  }

  std::string key = llvm::formatv("{0}_slice{1}", in_name, slice_idx).str();
  auto mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].vec_reload_addr[0].second.addr;
  if (map_store_to_load_value.find(input) != map_store_to_load_value.end()) {
    //load的value曾被store过，则使用第2次分配的lmem地址
    mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].vec_reload_addr[1].second.addr;
  }
  auto mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].size;
  attrs.push_back(builder.getNamedAttr(
      LocalGenInterface::kLayerGroupAttrName,
      getLgParam(ti, ts, mem_s_addr, mem_s_size, lgInfo.type, 0, 0, slice_idx, can_merge)));
  if (current_op_ == nullptr) {
    builder.setInsertionPointToStart(body_);
  } else {
    builder.setInsertionPointAfter(current_op_);
  }

  Type out_type;
  if (version == 0 || module::isWeight(input) || module::isDynWeight(input)) {
    out_type = input.getType();
  } else {
    int n = ti.slice_info.n[ncdhw_idx[0]].second;
    int c = ti.slice_info.c[ncdhw_idx[1]].second;
    // int d = ti.slice_info.d[ncdhw_idx[2]].second;
    int h = ti.slice_info.h[ncdhw_idx[3]].second;
    int w = ti.slice_info.w[ncdhw_idx[4]].second;
    out_type = RankedTensorType::get({n,c,h,w}, builder.getF32Type());
  }

  std::vector<Value> operands;
  if (map_old_grp_out_to_new_grp_out.find(input) != map_old_grp_out_to_new_grp_out.end()
   && std::find(lgInfo.group_ins.begin(), lgInfo.group_ins.end(), input) != lgInfo.group_ins.end()) {
    operands.push_back(map_old_grp_out_to_new_grp_out[input]);
  } else {
    // map_old_to_new_value for moveOp, wxc todo
    if (map_store_to_load_value.find(input) != map_store_to_load_value.end()) {
      auto store_op = dyn_cast<tpu::StoreOp>(map_store_to_load_value[input].getDefiningOp());
      if (isa<top::NoneOp>(store_op->getOperand(1).getDefiningOp())) {
        operands.push_back(store_op->getOperand(0));
      } else {
        operands.push_back(map_store_to_load_value[input]);
      }
    } else if (map_l2m_out_to_load_in.find(input) != map_l2m_out_to_load_in.end()){
      operands.push_back(map_l2m_out_to_load_in[input]);
    } else {
      operands.push_back(input);
    }
  }
  auto loadOp =
      builder.create<tpu::LoadOp>(NameLoc::get(builder.getStringAttr(name)),
                                  out_type, operands, attrs);
  map_old_to_new_value[input][slice_idx] = loadOp->getResult(0);
  current_op_ = loadOp;
}

Value GroupOps::CreateStoreOp2(Value &output, tensor_info_t& ti, int64_t ts, int64_t slice_idx, int64_t pipe_id,
                                group_type_t group_type, bool can_merge) {
  auto builder = OpBuilder(ctx_);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto new_value = map_old_to_new_value[output][slice_idx];
  operands.push_back(new_value);
  auto out_tensor_name = module::getName(output).str();
  std::string name = "store_" + out_tensor_name + "_pipe" +std::to_string(pipe_id) + "_slice" + std::to_string(slice_idx);
  LOG(INFO) <<"store_value: " << name;
  std::string key = llvm::formatv("{0}_slice{1}", out_tensor_name, slice_idx).str();
  auto mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].vec_reload_addr[0].second.addr;
  auto mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].size;
  attrs.push_back(builder.getNamedAttr(
      LocalGenInterface::kLayerGroupAttrName,
      getLgParam(ti, ts, mem_s_addr, mem_s_size, group_type, 0, 0, slice_idx, can_merge)));
  builder.setInsertionPointAfter(current_op_);

  Operation* storeOp = nullptr;
  // bool will_store = std::find(will_store_value[pipe_id].begin(), will_store_value[pipe_id].end(), output) != will_store_value[pipe_id].end();
  if (map_store_tensor_to_outbuffer_out.find(output) != map_store_tensor_to_outbuffer_out.end()) {
    operands.push_back(map_store_tensor_to_outbuffer_out[output]);
    storeOp =
        builder.create<tpu::StoreOp>(NameLoc::get(builder.getStringAttr(name)),
                                    output.getType(), operands, attrs);
  } else {
    auto noneOp = module::getNoneOp(output.getDefiningOp());
    operands.push_back(noneOp);
    storeOp =
        builder.create<tpu::StoreOp>(NameLoc::get(builder.getStringAttr(name)),
                                    output.getType(), operands, attrs);
  }
  if (std::find(need_store_load_value[pipe_id].begin(), need_store_load_value[pipe_id].end(), output)
                != need_store_load_value[pipe_id].end()) {
    map_store_to_load_value[output] = storeOp->getResult(0);
  }

  auto label = llvm::formatv("ts:{0}", ts).str();
  map_name_output_to_merge_slice_for_grp[output].push_back(name);
  current_op_ = storeOp;
  map_old_to_new_value[output][slice_idx] = storeOp->getResult(0);
  return storeOp->getResult(0);
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

void GroupOps::UpdateOpLgParam2(Operation *op, Operation *old_op, int64_t ts,  int64_t slice_idx,
                              TensorInfo &tensor_info, std::vector<int64_t> ncdhw_idx, group_type_t group_type, bool can_merge) {
  auto builder = OpBuilder(ctx_);
  auto output = *old_op->getResults().begin();
  std::string name = module::getName(output).str();
  auto &ti = tensor_info[output];
  if (version == 1) {
    int n = tensor_info[old_op->getResult(0)].slice_info.n[ncdhw_idx[0]].second;
    int c = tensor_info[old_op->getResult(0)].slice_info.c[ncdhw_idx[1]].second;
    int h = tensor_info[old_op->getResult(0)].slice_info.h[ncdhw_idx[3]].second;
    int w = tensor_info[old_op->getResult(0)].slice_info.w[ncdhw_idx[4]].second;
    auto out_type = RankedTensorType::get({n,c,h,w}, builder.getF32Type());
    op->getResult(0).setType(out_type); //todo maxpool indices,second output
  }
  std::string key = llvm::formatv("{0}_slice{1}", name, slice_idx).str();
  auto mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].vec_reload_addr[0].second.addr;
  auto mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].size;
  std::string imm_key = llvm::formatv("{0}_buffer_slice{1}", name, slice_idx).str();
  int imm_mem_s_addr = 0, imm_mem_s_size = 0;
  if (ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his.find(imm_key)
      != ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his.end()) {
    imm_mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[imm_key].vec_reload_addr[0].second.addr;
    imm_mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[imm_key].size;
  }
  std::vector<std::vector<int64_t>> opd_h_slice_offset;
  for (auto v : old_op->getOperands()) {
    opd_h_slice_offset.push_back(std::vector<int64_t>());
    auto slice_infos = tensor_info[v].slice_infos;
    if (slice_infos.size() > 1) {
      for (auto itr = slice_infos.begin(); itr != slice_infos.end(); ++itr) {
        if (itr->first == old_op) {
          auto si = itr->second.h;
          auto full_h_slice_info = tensor_info[v].slice_info.h;
          for (int i = 0; i < si.size(); i++) {
            assert(full_h_slice_info[i].first <= si[i].first);
            opd_h_slice_offset.back().push_back(si[i].first);
          }
          break;
        }
      }
    }
  }
  op->setAttr(LocalGenInterface::kLayerGroupAttrName,
              getLgParam(ti, ts, mem_s_addr, mem_s_size, group_type, imm_mem_s_addr, imm_mem_s_size,
                        slice_idx, can_merge, opd_h_slice_offset));
}

LayerGroupAttr GroupOps::getLgParam(tensor_info_t &tensor_info, int64_t id,
                                    int64_t out_addr, int64_t out_size,
                                    int64_t group_type, int64_t buffer_addr,
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
  for (auto itr: opd_h_slice_offset) {
    for (auto itr2: itr) {
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
