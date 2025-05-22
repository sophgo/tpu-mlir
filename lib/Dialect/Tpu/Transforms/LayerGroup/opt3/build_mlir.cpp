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

using namespace tpu_mlir::tpu;
using namespace tpu_mlir::backend;

typedef struct node_info {
  Operation *global_op = nullptr;
  LgInfo *lgInfo = nullptr;
  int indeg = 0;
  int idx = 0;
  std::vector<node_info *> pre_nodes;
  std::vector<node_info *> next_nodes;
  std::vector<node_info *> tmp_pre_nodes;
  node_info(Operation *global_op_) : global_op(global_op_) {}
  node_info(LgInfo *lgInfo_) : lgInfo(lgInfo_) {}
  bool operator==(const node_info &rhs) const {
    return global_op == rhs.global_op && lgInfo == rhs.lgInfo;
  }

  std::string name() {
    if (global_op) {
      return module::getName(global_op).str();
    } else {
      std::string tmp_str = "";
      int i = 0;
      for (auto it : lgInfo->group_ops) {
        if (!it)
          continue;
        if (i++ > 5) {
          break;
        }
        tmp_str = tmp_str + "---" + module::getName(it).str();
      }
      assert(tmp_str.size() > 0);
      return tmp_str.substr(3, tmp_str.size());
    }
  }

  std::string first_op_name() {
    if (global_op) {
      return module::getName(global_op).str();
    } else {
      return module::getName(lgInfo->group_ops[0]).str();
    }
  }

  void show_info(std::string extra_info = "I am") {
    if (!module::isDebugCmdEnable("print_node_topo_change")) {
      return;
    }
    if (global_op) {
      LOG(INFO) << extra_info << " at global_op: " << name();
    } else {
      LOG(INFO) << extra_info << " at group:" << name();
    }
  }
} node_info;

bool node_info_Sort_by_int(const node_info &v1, const node_info &v2) {
  return v1.idx < v2.idx; // Ascending arrangement
}

static void nodes_topo_order_dfs(node_info &cur_node,
                                 std::vector<node_info> &topo_nodes, int &idx) {
  assert(cur_node.global_op || cur_node.lgInfo);
  cur_node.idx = idx++;
  topo_nodes.push_back(cur_node);
  cur_node.show_info("add into topo_nodes, idx:" + std::to_string(idx - 1));
  for (auto &next_node : cur_node.next_nodes) {
    if (next_node->indeg > 0) {
      next_node->indeg -= 1;
    }
    auto &nodes = next_node->tmp_pre_nodes;
    nodes.erase(std::remove(nodes.begin(), nodes.end(), &cur_node),
                nodes.end());
    next_node->show_info("check next_node, indeg:" +
                         std::to_string(next_node->indeg));
    if (next_node->indeg == 0) {
      if (std::find(topo_nodes.begin(), topo_nodes.end(), *next_node) ==
          topo_nodes.end()) {
        nodes_topo_order_dfs(*next_node, topo_nodes, idx);
      }
    }
  }
}

void GroupOps::CreateLmemMoveOp(int64_t ts, ts_move_info &move_info) {
  assert(current_op_ != nullptr);
  auto builder = OpBuilder(ctx_);
  builder.setInsertionPointAfter(current_op_);

  std::vector<Value> operands;
  int i = 0;
  for (auto itr : move_info.move_value) {
    LOG(INFO) << "need move value:" << module::getName(itr).str();
    auto new_value = map_old_to_new_value[itr][move_info.slice_idx[i]];
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
  attrs.push_back(builder.getNamedAttr("ts_id", builder.getI64IntegerAttr(ts)));
  attrs.push_back(
      builder.getNamedAttr("name", builder.getStringAttr(move_info.name)));
  auto op_loc = builder.getFusedLoc(locs);
  auto moveOp = builder.create<tpu::MoveOp>(op_loc, ret_types, operands, attrs);

  i = 0;
  for (auto itr : move_info.move_value) {
    map_old_to_new_value[itr][move_info.slice_idx[i]] = moveOp->getResult(i);
    // LOG(INFO) <<"xxxxx12, name:"<<itr.name<<",
    // value:"<<module::getName(itr.value).str()<<", slice:"<<itr.slice_idx;
    i++;
  }
  current_op_ = moveOp;
}

void GroupOps::CreateLoadToL2mOp(int64_t ts, l2m_value_info &it,
                                 int64_t pipe_id,
                                 l2mem_alloc_Ptr l2mem_alloc_ptr) {
  Value &input = it.value;
  std::string in_name = module::getName(input).str();
  int64_t slice_idx = it.slice_idx;
  if (!it.valid) {
    llvm::errs() << "CreateLoadToL2mOp, name:" << in_name
                 << ", slice_idx:" << slice_idx << "invalid, skiped\n";
    return;
  }
  auto builder = OpBuilder(ctx_);
  std::vector<NamedAttribute> attrs;
  std::string name = "loadToL2m_";
  name = name + in_name;
  name = name + "_pipe" + std::to_string(pipe_id) + "_slice" +
         std::to_string(slice_idx);

  std::string key = llvm::formatv("{0}_slice{1}", in_name, slice_idx).str();
  if (l2mem_alloc_ptr->vec_mem_alloc_his.find(key) ==
      l2mem_alloc_ptr->vec_mem_alloc_his.end()) {
    key = llvm::formatv("{0}_slice-1", in_name).str();
  }
  llvm::errs() << " load value:" << name << ", key:" << key << "\n";
  auto buffer_value =
      l2mem_alloc_ptr->vec_mem_alloc_his[key].vec_reload_addr[0].second.buffer;

  attrs.push_back(builder.getNamedAttr("id", builder.getI64IntegerAttr(ts)));
  if (current_op_ == nullptr) {
    builder.setInsertionPointToStart(body_);
  } else {
    builder.setInsertionPointAfter(current_op_);
  }

  std::vector<Value> operands;
  if (map_old_grp_out_to_new_grp_out.find(input) !=
      map_old_grp_out_to_new_grp_out.end()) {
    operands.push_back(map_old_grp_out_to_new_grp_out[input]);
  } else {
    operands.push_back(input);
  }
  operands.push_back(buffer_value);
  auto loadToL2mOp = builder.create<tpu::LoadToL2MOp>(
      NameLoc::get(builder.getStringAttr(name)), input.getType(), operands,
      attrs);
  map_l2m_out_to_load_in[input] = loadToL2mOp->getResult(0);
  current_op_ = loadToL2mOp;
}

void GroupOps::CreateLoadOp2(
    int64_t ts, ts_var_t &ts_var, int64_t pipe_id,
    const std::vector<Operation *> &ops, std::vector<int64_t> ncdhw_idx,
    const LgInfo &lgInfo, bool can_merge,
    std::map<Value, Value, value_compare> &map_old_v_to_new_v_in_group_in) {
  Value &input = ts_var.value;
  tensor_info_t &ti = ts_var.info;
  int64_t slice_idx = ts_var.slice_idx;
  auto builder = OpBuilder(ctx_);
  auto inputOp = input.getDefiningOp();
  std::vector<NamedAttribute> attrs;
  bool train = true;
  std::string name = "load_";
  std::string in_name = module::getName(input).str();
  LOG(INFO) << "load_value: " << in_name;
  if (inputOp != nullptr) {
    name = name + in_name;
  } else {
    int arg_idx = input.cast<BlockArgument>().getArgNumber();
    name = name + std::to_string(arg_idx);
  }
  name = name + "_pipe" + std::to_string(pipe_id) + "_slice" +
         std::to_string(slice_idx);
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
  auto mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key]
                        .vec_reload_addr[0]
                        .second.addr;
  if (map_store_to_load_value.find(input) != map_store_to_load_value.end()) {
    // If the value of the load is previously stored, the lmem address assigned
    // for the second time is used
    mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key]
                     .vec_reload_addr[1]
                     .second.addr;
  }
  auto mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].size;
  attrs.push_back(builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                                       getLgParam(ti, ts, mem_s_addr,
                                                  mem_s_size, lgInfo.type, 0, 0,
                                                  slice_idx, can_merge)));
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
    out_type = RankedTensorType::get({n, c, h, w}, builder.getF32Type());
  }

  std::vector<Value> operands;
  if (map_old_grp_out_to_new_grp_out.find(input) !=
          map_old_grp_out_to_new_grp_out.end() &&
      std::find(lgInfo.group_ins.begin(), lgInfo.group_ins.end(), input) !=
          lgInfo.group_ins.end()) {
    operands.push_back(map_old_grp_out_to_new_grp_out[input]);
  } else {
    // map_old_to_new_value for moveOp, wxc todo
    if (map_store_to_load_value.find(input) != map_store_to_load_value.end()) {
      auto store_op = dyn_cast<tpu::StoreOp>(
          map_store_to_load_value[input].getDefiningOp());
      if (isa<top::NoneOp>(store_op->getOperand(1).getDefiningOp())) {
        operands.push_back(store_op->getOperand(0));
      } else {
        operands.push_back(map_store_to_load_value[input]);
      }
    } else if (map_l2m_out_to_load_in.find(input) !=
               map_l2m_out_to_load_in.end()) {
      operands.push_back(map_l2m_out_to_load_in[input]);
    } else {
      operands.push_back(input);
    }
  }
  // for add op from mlp group part sum
  if (map_old_v_to_new_v_in_group_in.find(input) !=
      map_old_v_to_new_v_in_group_in.end()) {
    input = map_old_v_to_new_v_in_group_in[input];
    inputOp = input.getDefiningOp();
    operands.clear();
    operands.push_back(input);
  }
  auto loadOp = builder.create<tpu::LoadOp>(
      NameLoc::get(builder.getStringAttr(name)), out_type, operands, attrs);
  map_old_to_new_value[input][slice_idx] = loadOp->getResult(0);
  current_op_ = loadOp;
}

Value GroupOps::CreateStoreOp2(Value &output, tensor_info_t &ti, int64_t ts,
                               int64_t slice_idx, int64_t pipe_id,
                               group_type_t group_type, bool can_merge,
                               l2mem_alloc_Ptr l2mem_alloc_ptr) {
  auto builder = OpBuilder(ctx_);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto new_value = map_old_to_new_value[output][slice_idx];
  operands.push_back(new_value);
  auto out_tensor_name = module::getName(output).str();
  std::string name = "store_" + out_tensor_name + "_pipe" +
                     std::to_string(pipe_id) + "_slice" +
                     std::to_string(slice_idx);
  LOG(INFO) << "store_value: " << name;
  std::string key =
      llvm::formatv("{0}_slice{1}", out_tensor_name, slice_idx).str();
  auto mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key]
                        .vec_reload_addr[0]
                        .second.addr;
  auto mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].size;
  attrs.push_back(
      builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                           getLgParam(ti, ts, mem_s_addr, mem_s_size,
                                      group_type, 0, 0, slice_idx, can_merge)));

  key = llvm::formatv("{0}_slice0", out_tensor_name).str();
  builder.setInsertionPointAfter(current_op_);

  Operation *storeOp = nullptr;
  // bool will_store = std::find(will_store_value[pipe_id].begin(),
  // will_store_value[pipe_id].end(), output) !=
  // will_store_value[pipe_id].end();
  if (map_store_tensor_to_outbuffer_out.find(output) !=
      map_store_tensor_to_outbuffer_out.end()) {
    operands.push_back(map_store_tensor_to_outbuffer_out[output]);
  } else if (l2mem_alloc_ptr && l2mem_alloc_ptr->vec_mem_alloc_his.find(key) !=
                                    l2mem_alloc_ptr->vec_mem_alloc_his.end()) {
    auto buffer_value = l2mem_alloc_ptr->vec_mem_alloc_his[key]
                            .vec_reload_addr[0]
                            .second.buffer;
    operands.push_back(buffer_value);
  } else {
    operands.push_back(none_op_->getResult(0));
  }
  storeOp =
      builder.create<tpu::StoreOp>(NameLoc::get(builder.getStringAttr(name)),
                                   output.getType(), operands, attrs);
  if (std::find(need_store_load_value[pipe_id].begin(),
                need_store_load_value[pipe_id].end(),
                output) != need_store_load_value[pipe_id].end()) {
    map_store_to_load_value[output] = storeOp->getResult(0);
  }

  auto label = llvm::formatv("ts:{0}", ts).str();
  map_name_output_to_merge_slice_for_grp[output].push_back(name);
  current_op_ = storeOp;
  map_old_to_new_value[output][slice_idx] = storeOp->getResult(0);
  return storeOp->getResult(0);
}

void GroupOps::UpdateOpLgParam2(Operation *op, Operation *old_op, int64_t ts,
                                int64_t slice_idx, TensorInfo &tensor_info,
                                std::vector<int64_t> ncdhw_idx,
                                group_type_t group_type, bool can_merge) {
  auto builder = OpBuilder(ctx_);
  auto output = *old_op->getResults().begin();
  std::string name = module::getName(output).str();
  auto &ti = tensor_info[output];
  if (version == 1) {
    int n = tensor_info[old_op->getResult(0)].slice_info.n[ncdhw_idx[0]].second;
    int c = tensor_info[old_op->getResult(0)].slice_info.c[ncdhw_idx[1]].second;
    int h = tensor_info[old_op->getResult(0)].slice_info.h[ncdhw_idx[3]].second;
    int w = tensor_info[old_op->getResult(0)].slice_info.w[ncdhw_idx[4]].second;
    auto out_type = RankedTensorType::get({n, c, h, w}, builder.getF32Type());
    op->getResult(0).setType(out_type); // todo maxpool indices,second output
  }
  std::string key = llvm::formatv("{0}_slice{1}", name, slice_idx).str();
  auto mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key]
                        .vec_reload_addr[0]
                        .second.addr;
  auto mem_s_size = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[key].size;
  std::string imm_key =
      llvm::formatv("{0}_buffer_slice{1}", name, slice_idx).str();
  int imm_mem_s_addr = 0, imm_mem_s_size = 0;
  if (ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his.find(imm_key) !=
      ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his.end()) {
    imm_mem_s_addr = ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[imm_key]
                         .vec_reload_addr[0]
                         .second.addr;
    imm_mem_s_size =
        ILP_time_step->lmem_alloc_ptr->vec_mem_alloc_his[imm_key].size;
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
              getLgParam(ti, ts, mem_s_addr, mem_s_size, group_type,
                         imm_mem_s_addr, imm_mem_s_size, slice_idx, can_merge,
                         opd_h_slice_offset));
}

static void BFSPrintNodeGraph(std::shared_ptr<dot_graph> dot_graph_log,
                              node_info *start_node, bool fwd = true,
                              int depth = 8) {
  if (depth == 0) {
    return;
  }
  int tmp_depth = depth - 1;
  dot_graph_log->add_node_into_graph(start_node->name());
  if (fwd) {
    for (auto node : start_node->next_nodes) {
      dot_graph_log->add_node_into_graph(node->name());
      dot_graph_log->add_edge_into_graph(start_node->name(), node->name());
      llvm::errs() << "depth: " << tmp_depth << ", next_node: " << node->name()
                   << "\n";
      BFSPrintNodeGraph(dot_graph_log, node, fwd, tmp_depth);
    }
  } else {
    for (auto pre_node : start_node->pre_nodes) {
      dot_graph_log->add_node_into_graph(pre_node->name());
      dot_graph_log->add_edge_into_graph(pre_node->name(), start_node->name());
      llvm::errs() << "depth: " << tmp_depth
                   << ", pre_node: " << pre_node->name() << "\n";
      BFSPrintNodeGraph(dot_graph_log, pre_node, fwd, tmp_depth);
    }
  }
}

void export_node_subnet_dot(std::vector<node_info> &nodes) {
  if (module::isDebugCmdEnable("export_node_subnet_dot")) {
    for (auto node : nodes) {
      if (module::isDebugCmdEnable("export_node_subnet_dot-" +
                                   node.first_op_name() + "-")) {
        auto dot_graph_log = std::make_shared<dot_graph>();
        BFSPrintNodeGraph(dot_graph_log, &node, false, 8);
        BFSPrintNodeGraph(dot_graph_log, &node, true, 8);
        dot_graph_log->export_dot("export_node_subnet_dot-" +
                                  node.first_op_name());
        break;
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

  bool print_node_topo_change =
      module::isDebugCmdEnable("print_node_topo_change");
  std::vector<node_info> nodes;
  std::vector<Operation *> all_local_ops;
  std::map<node_info *, std::vector<node_info *>> map_parallel_node_subnet;
  for (int64_t i = 0; i < group_num; i++) {
    if (lg_infos[i].group_ops.size() > 1) {
      node_info tmp(&lg_infos[i]);
      tmp.show_info("add group");
      nodes.push_back(tmp);
      all_local_ops.insert(all_local_ops.end(), lg_infos[i].group_ops.begin(),
                           lg_infos[i].group_ops.end());
      if (print_node_topo_change) {
        for (auto op : lg_infos[i].group_ops) {
          if (op)
            llvm::errs() << "op: " << module::getName(op).str() << "\n";
        }
        for (auto out : lg_infos[i].group_outs) {
          llvm::errs() << "   group_outs: " << module::getName(out).str()
                       << "\n";
        }
        for (auto in : lg_infos[i].group_ins) {
          llvm::errs() << "   group_ins: " << module::getName(in).str() << "\n";
        }
      }
    }
  }

  //添加global node
  func_.walk([&](Operation *op) {
    if (!isa<FuncOp, ReturnOp, top::NoneOp>(op)) {
      if (std::find(all_local_ops.begin(), all_local_ops.end(), op) ==
          all_local_ops.end()) {
        if (print_node_topo_change) {
          LOG(INFO) << "add global_op: " << module::getName(op).str();
        }
        nodes.push_back(node_info(op));
      }
    }
  });
  Operation *retrunOp = nullptr;
  func_.walk([&](Operation *op) {
    if (!retrunOp && isa<ReturnOp>(op)) {
      retrunOp = op;
    }
    if (!none_op_ && isa<top::NoneOp>(op)) {
      LOG(INFO) << "find NoneOp";
      none_op_ = op;
    }
  });

  if (!none_op_) {
    none_op_ = module::getNoneOp(lg_infos[0].group_ops[0]);
  }

  //获取node的前后驱node
  LOG(INFO) << "get pre&next nodes, nodes.size: " << nodes.size();
  for (auto &node : nodes) {
    if (node.global_op == nullptr) {
      node.show_info();
      for (auto in : node.lgInfo->group_ins) {
        auto in_op = in.getDefiningOp();
        if (in_op && !isa<top::NoneOp>(in_op)) {
          for (auto &node2 : nodes) {
            if (node == node2)
              continue;
            if (node2.global_op) {
              if (in_op == node2.global_op) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(),
                              &node2) == node.pre_nodes.end()) {
                  if (print_node_topo_change) {
                    LOG(INFO)
                        << "find pre node: " << module::getName(in_op).str();
                  }
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            } else {
              auto &node_ops = node2.lgInfo->group_ops;
              if (std::find(node_ops.begin(), node_ops.end(), in_op) !=
                  node_ops.end()) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(),
                              &node2) == node.pre_nodes.end()) {
                  if (print_node_topo_change) {
                    LOG(INFO)
                        << "find pre group: " << module::getName(in_op).str();
                  }
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            }
          }
        }
      }
      for (auto out : node.lgInfo->group_outs) {
        for (auto user : out.getUsers()) {
          if (isa<ReturnOp>(user)) {
            continue;
          }
          for (auto &node2 : nodes) {
            if (node == node2)
              continue;
            if (node2.global_op) {
              if (user == node2.global_op) {
                if (std::find(node.next_nodes.begin(), node.next_nodes.end(),
                              &node2) == node.next_nodes.end()) {
                  if (print_node_topo_change) {
                    LOG(INFO)
                        << "find next node: " << module::getName(user).str();
                  }
                  node.next_nodes.push_back(&node2);
                  break;
                }
              }
            } else {
              auto &node_ops = node2.lgInfo->group_ops;
              if (std::find(node_ops.begin(), node_ops.end(), user) !=
                  node_ops.end()) {
                if (std::find(node.next_nodes.begin(), node.next_nodes.end(),
                              &node2) == node.next_nodes.end()) {
                  if (print_node_topo_change) {
                    LOG(INFO) << "find next group, have:"
                              << module::getName(user).str();
                  }
                  node.next_nodes.push_back(&node2);
                  break;
                }
              }
            }
          }
        }
      }
    } else {
      if (print_node_topo_change) {
        LOG(INFO) << "check node: " << module::getName(node.global_op).str();
      }
      node.show_info();
      for (auto in : node.global_op->getOperands()) {
        auto in_op = in.getDefiningOp();
        if (in_op && !isa<top::NoneOp>(in_op)) {
          for (auto &node2 : nodes) {
            if (node == node2)
              continue;
            if (node2.global_op) {
              if (in_op == node2.global_op) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(),
                              &node2) == node.pre_nodes.end()) {
                  if (print_node_topo_change) {
                    LOG(INFO)
                        << "find pre node: " << module::getName(in_op).str();
                  }
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            } else {
              auto &node_ops = node2.lgInfo->group_ops;
              if (std::find(node_ops.begin(), node_ops.end(), in_op) !=
                  node_ops.end()) {
                if (std::find(node.pre_nodes.begin(), node.pre_nodes.end(),
                              &node2) == node.pre_nodes.end()) {
                  if (print_node_topo_change) {
                    LOG(INFO)
                        << "find pre group: " << module::getName(in_op).str();
                  }
                  node.pre_nodes.push_back(&node2);
                  break;
                }
              }
            }
          }
        }
      }
      for (auto user : node.global_op->getUsers()) {
        if (isa<ReturnOp>(user)) {
          continue;
        }
        for (auto &node2 : nodes) {
          if (node == node2)
            continue;
          if (node2.global_op) {
            if (user == node2.global_op) {
              if (std::find(node.next_nodes.begin(), node.next_nodes.end(),
                            &node2) == node.next_nodes.end()) {
                if (print_node_topo_change) {
                  LOG(INFO)
                      << "find next node: " << module::getName(user).str();
                }
                node.next_nodes.push_back(&node2);
                break;
              }
            }
          } else {
            auto &node_ops = node2.lgInfo->group_ops;
            if (std::find(node_ops.begin(), node_ops.end(), user) !=
                node_ops.end()) {
              if (std::find(node.next_nodes.begin(), node.next_nodes.end(),
                            &node2) == node.next_nodes.end()) {
                if (print_node_topo_change) {
                  LOG(INFO) << "find next group, have:"
                            << module::getName(user).str();
                }
                node.next_nodes.push_back(&node2);
                break;
              }
            }
          }
        }
      }
    }
  }
  LOG(INFO) << "nodes.size: " << nodes.size();
  for (auto &node : nodes) {
    node.indeg = node.pre_nodes.size();
    node.tmp_pre_nodes.assign(node.pre_nodes.begin(), node.pre_nodes.end());
  }
  int idx = 0;
  std::vector<node_info> topo_nodes;
  for (auto &node : nodes) {
    if (node.indeg == 0) {
      if (std::find(topo_nodes.begin(), topo_nodes.end(), node) ==
          topo_nodes.end()) {
        node.show_info("dfs start point");
        nodes_topo_order_dfs(node, topo_nodes, idx);
      }
    }
  }

  // lg_pass_ir_->map_parallel_node_subnet

  std::sort(topo_nodes.begin(), topo_nodes.end(), node_info_Sort_by_int);
  LOG(INFO) << "topo_nodes.size: " << topo_nodes.size();
  if (print_node_topo_change) {
    for (auto &node : nodes) {
      if (node.tmp_pre_nodes.size() > 0) {
        node.show_info("have untrack pre_nodes:");
        for (auto node2 : node.tmp_pre_nodes) {
          node2->show_info("  ");
        }
      }
    }
  }
  export_node_subnet_dot(nodes);

  typedef struct lg_extra_info {
    std::vector<Operation *> slice_merge_ops;
    std::vector<Operation *> grp_group_ops;
    std::vector<Operation *> outbuffer_ops;
    std::vector<Operation *> buffer_ops;
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
    if (lg_infos[group_idx].group_ops.size() > 1 &&
        lg_pass_ir_->ILP_time_steps[group_idx].size() > 0) {
      map_name_output_to_merge_slice_for_grp.clear();
      std::map<Value, std::vector<Value>, value_compare>
          map_output_to_merge_slice_for_grp;
      map_store_tensor_to_outbuffer_out.clear();
      const LgInfo &lg_info = lg_infos[group_idx];
      TensorInfo &tensor_info = lg_pass_ir_->lg_tensor_infos_[group_idx];
      auto &ops = lg_info.group_ops;
      auto &op_outs = lg_info.group_op_outs;
      for (auto it0 = ops.rbegin(); it0 != ops.rend(); ++it0) {
        current_op_ = *it0;
        if (current_op_)
          break;
      }
      need_store_load_value.clear();
      will_store_value.clear();
      std::vector<Value> need_dump_tensor_values;
      for (int pipe_id = 0;
           pipe_id < lg_pass_ir_->ILP_time_steps[group_idx].size(); pipe_id++) {
        LOG(INFO) << "preprocess grp: " << group_idx
                  << ", unique pipe_id:" << pipe_id;
        ILP_time_step = lg_pass_ir_->ILP_time_steps[group_idx][pipe_id];
        int ts_count = ILP_time_step->ts_count;
        std::vector<Value> tmp_need_store_load_value;
        for (size_t ts = 0; ts < ts_count; ++ts) {
          LOG(INFO) << "----------------ts" << ts << "-----------------";
          for (auto it : ILP_time_step->timestep_table_new[ts].vec_ts_var) {
            if (it.var_value == 1) {
              auto itr = ILP_time_step->values_need_store_to_grpout.begin();
              for (; itr != ILP_time_step->values_need_store_to_grpout.end();
                   ++itr) {
                // if (it.varName == itr->second) { //new edit
                if (std::find(itr->second.begin(), itr->second.end(),
                              it.varName) != itr->second.end()) {
                  tmp_need_store_load_value.push_back(itr->first);
                  if (map_store_tensor_to_outbuffer_out.find(itr->first) ==
                      map_store_tensor_to_outbuffer_out.end()) {
                    assert(itr->first == it.value);
                    builder.setInsertionPointAfter(current_op_);
                    auto loc = NameLoc::get(builder.getStringAttr(
                        module::getName(itr->first).str()));
                    auto out_grp_op = builder.create<tpu::OutBufferOp>(
                        loc, itr->first.getType(), ValueRange{});
                    map_lg_extra_info[(int64_t)&lg_info]
                        .outbuffer_ops.push_back(out_grp_op);
                    map_store_tensor_to_outbuffer_out[itr->first] =
                        out_grp_op.getResult();
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
          LOG(INFO) << "----------------collect will_store_value at ts" << ts
                    << "-----------------";
          for (auto it : ILP_time_step->timestep_table_new[ts].vec_ts_var) {
            if (it.var_value == 1) {
              auto out_name = module::getName(it.value).str();
              if (it.info.mode2 & TIMESTEP2_STORE) {
                LOG(INFO) << "add into will_store_value for store, name:"
                          << out_name;
                tmp_will_store_value.push_back(it.value);
              } else if (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD &&
                         ILP_time_step->mapILPVarInfo[it.varName]
                                 .store_load_mode == 0) {
                LOG(INFO)
                    << "add into will_store_value for store_and_load, name:"
                    << out_name;
                tmp_will_store_value.push_back(it.value);
              }
            }
          }
        }
        will_store_value.push_back(tmp_will_store_value);

        for (auto op_out : op_outs) {
          if (need_dump_tensors[0] == "not_dump" ||
              std::find(ops.begin(), ops.end(), op_out.getDefiningOp()) ==
                  ops.end()) {
            continue;
          }
          auto out_name = module::getName(op_out).str();
          bool not_need_create =
              std::find(tmp_will_store_value.begin(),
                        tmp_will_store_value.end(),
                        op_out) != tmp_will_store_value.end();
          if (need_dump_tensors[0] == "dump_all" ||
              std::find(need_dump_tensors.begin(), need_dump_tensors.end(),
                        out_name) != need_dump_tensors.end()) {
            if (!not_need_create) {
              tmp_need_store_load_value.push_back(op_out);
            }
            if (map_store_tensor_to_outbuffer_out.find(op_out) ==
                map_store_tensor_to_outbuffer_out.end()) {
              if (!not_need_create) {
                need_dump_tensor_values.push_back(op_out);
                builder.setInsertionPointAfter(current_op_);
                auto loc = NameLoc::get(builder.getStringAttr(out_name));
                auto out_grp_op = builder.create<tpu::OutBufferOp>(
                    loc, op_out.getType(), builder.getBoolAttr(true));
                map_lg_extra_info[(int64_t)&lg_info].outbuffer_ops.push_back(
                    out_grp_op);
                map_store_tensor_to_outbuffer_out[op_out] =
                    out_grp_op.getResult();
                current_op_ = out_grp_op;
                // LOG(INFO) <<"out_grp_op.dump2:";
                // out_grp_op.dump();
              }
            } else {
              auto whOp =
                  map_store_tensor_to_outbuffer_out[op_out].getDefiningOp();
              llvm::errs() << "change need_dump for " << module::getName(op_out)
                           << "\n";
              whOp->setAttr("need_dump", builder.getBoolAttr(true));
            }
          }
        }
        llvm::errs() << "tmp_need_store_load_value.size:"
                     << tmp_need_store_load_value.size() << "\n";
        need_store_load_value.push_back(tmp_need_store_load_value);
      }

      for (auto out : lg_info.group_outs) {
        map_output_to_merge_slice_for_grp[out] = std::vector<Value>();
        map_name_output_to_merge_slice_for_grp[out] =
            std::vector<std::string>();
      }
      for (auto out : lg_info.group_op_outs) {
        map_old_to_new_value[out] = std::map<int, Value>();
      }
      bool one_grp = lg_pass_ir_->ILP_time_steps[group_idx].size() == 1;

      llvm::errs() << "group" << group_idx << ", group_id:" << lg_info.group_id
                   << ", in and out:\n";
      for (auto op : lg_info.group_ops) {
        if (op)
          llvm::errs() << "  op:" << show_op_info(op) << "\n";
      }
      for (auto out : lg_info.group_outs) {
        llvm::errs() << "    out:" << module::getName(out).str() << "\n";
      }
      for (auto in : lg_info.group_ins) {
        llvm::errs() << "    in:" << module::getName(in).str() << "\n";
      }

      std::vector<Value> buffer_values;
      auto &l2mem_alloc_ptr = lg_pass_ir_->lg_l2mem_alloc_ptr[group_idx];
      int idx = 0;
      builder.setInsertionPointAfter(current_op_);
      for (auto &itr : l2mem_alloc_ptr->vec_mem_alloc_his) {
        auto type = ::mlir::Builder(ctx_).getIntegerType(8);
        auto out_type = RankedTensorType::get({(int64_t)itr.second.size}, type);
        auto tmpStr = "buffer_for_group_idx" + std::to_string(group_idx) +
                      "_idx" + std::to_string(idx++);
        auto newName = NameLoc::get(builder.getStringAttr(tmpStr));
        std::vector<NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            "buffer_type",
            tpu::BufferTypeAttr::get(ctx_, tpu::BufferType::L2)));
        auto newOp = builder.create<tpu::BufferOp>(newName, out_type,
                                                   ValueRange{}, attrs);
        current_op_ = newOp.getResult().getDefiningOp();
        map_lg_extra_info[(int64_t)&lg_info].buffer_ops.push_back(current_op_);
        itr.second.vec_reload_addr[0].second.buffer = newOp.getResult();
        buffer_values.push_back(newOp.getResult());
      }

      assert(lg_pass_ir_->ILP_time_steps[group_idx].size() > 0);
      for (int pipe_id = 0;
           pipe_id < lg_pass_ir_->ILP_time_steps[group_idx].size(); pipe_id++) {
        LOG(INFO) << "grp: " << group_idx << ", unique pipe_id:" << pipe_id;
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
          if (map_old_grp_out_to_new_grp_out.find(in) !=
              map_old_grp_out_to_new_grp_out.end()) {
            operands.push_back(map_old_grp_out_to_new_grp_out[in]);
          } else {
            operands.push_back(in);
          }
        }
        for (auto in : need_store_load_value[pipe_id]) {
          if (map_store_tensor_to_outbuffer_out.find(in) !=
              map_store_tensor_to_outbuffer_out.end()) {
            auto outbuffer_out = map_store_tensor_to_outbuffer_out[in];
            in_types.push_back(outbuffer_out.getType());
            operands.push_back(outbuffer_out);
          }
        }
        for (auto in : buffer_values) {
          in_types.push_back(in.getType());
          operands.push_back(in);
        }
        std::map<Value, std::vector<Value>, value_compare>
            map_output_to_merge_slice;
        std::map<Value, Value, value_compare> map_group_out_to_yield_in;
        llvm::SmallVector<NameLoc, 80> nameLocs;
        llvm::SmallVector<Location, 80> locs;
        for (auto out : lg_info.group_outs) {
          ret_types.push_back(out.getType());
          auto tmpStr = module::getLoc(out).getName().str();
          if (!one_grp) {
            tmpStr = tmpStr + "_pipe" + std::to_string(pipe_id);
          }
          LOG(INFO) << " new group_out name:" << tmpStr;
          auto newName = NameLoc::get(builder.getStringAttr(tmpStr));
          locs.push_back(newName);
          nameLocs.push_back(newName);
          outputs.push_back(out);
          map_output_to_merge_slice[out] = std::vector<Value>();
        }
        std::vector<int64_t> core_ids;
        std::vector<int64_t> core_slice_ncdhws;
        for (const auto &pair : ILP_time_step->ncdhw_steps) {
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
        attrs.push_back(builder.getNamedAttr("swpipl_stage_num",
                                             builder.getI64IntegerAttr(3)));
        attrs.push_back(builder.getNamedAttr(
            "run_core_id", builder.getI64ArrayAttr(core_ids)));
        attrs.push_back(builder.getNamedAttr(
            "core_slice_ncdhw", builder.getI64ArrayAttr(core_slice_ncdhws)));
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
            // The old grp outputs are replaced with new groupOp results, and
            // these grp outputs are converted to OpOperand whose owners are not
            // in the grp, i.e. the opd of the out-of-group user is replaced
            outputs[it.index()].replaceUsesWithIf(
                it.value(), [&](OpOperand &operand) {
                  Operation *user = operand.getOwner();
                  return find(ops.begin(), ops.end(), user) == ops.end();
                });
            map_old_grp_out_to_new_grp_out[outputs[it.index()]] = it.value();
            map_new_grp_out_to_old_grp_out[it.value()] = outputs[it.index()];
          } else {
            map_output_to_merge_slice_for_grp[outputs[it.index()]].push_back(
                it.value());
          }
        }

        current_op_ = nullptr;
        map_store_to_load_value.clear();
        map_l2m_out_to_load_in.clear();
        int ts_count = ILP_time_step->ts_count;
        bool one_slice = ILP_time_step->slice_num == 1;

        for (size_t ts = 0, ts_id = 0; ts < ts_count; ts++, ts_id++) {
          if (lg_pass_ir_->map_l2m_loads.size() > 0) {
            auto &map_l2m_load = lg_pass_ir_->map_l2m_loads[group_idx];
            // The start time slot must end as soon as possible, so the
            // LoadToL2M op cannot be inserted
            if (ts == 0) {
              if (map_l2m_load.find(-1) != map_l2m_load.end()) {
                LOG(INFO) << "----------------ts" << ts_id
                          << " for CreateLoadToL2mOp-----------------";
                for (auto it : map_l2m_load[-1]) {
                  CreateLoadToL2mOp(ts_id, it, pipe_id, l2mem_alloc_ptr);
                }
                ts_id++;
              }
            }
            if (ts >= 0) {
              if (map_l2m_load.find(ts) != map_l2m_load.end()) {
                LOG(INFO) << "----------------ts" << ts_id
                          << " for CreateLoadToL2mOp-----------------";
                for (auto it : map_l2m_load[ts]) {
                  CreateLoadToL2mOp(ts_id, it, pipe_id, l2mem_alloc_ptr);
                }
              }
            }
          }
          std::vector<ts_move_info> move_info;
          bool can_merge = ILP_time_step->timestep_table_new[ts].can_merge;
          auto &timeStep_table_ = ILP_time_step->inserted_timestep_table_;
          if (timeStep_table_.find(ts) != timeStep_table_.end()) {
            LOG(INFO) << "----------------ts" << ts_id
                      << " for CreateLmemMoveOp-----------------";
            move_info.assign(timeStep_table_[ts].begin(),
                             timeStep_table_[ts].end());
            if (move_info[0].combine_ts_op_idx == 0) {
              CreateLmemMoveOp(ts_id, move_info[0]);
              ts_id++;
            }
          }
          LOG(INFO) << "----------------ts" << ts_id << "-----------------";
          for (auto it : ILP_time_step->timestep_table_new[ts].vec_ts_var) {
            if (it.var_value == 1) {
              std::vector<int64_t> ncdhw_idx =
                  ILP_time_step->ncdhw_steps.begin()->second[it.slice_idx];
              if (it.info.mode2 & TIMESTEP2_LOAD) {
                CreateLoadOp2(ts_id, it, pipe_id, ops, ncdhw_idx, lg_info,
                              can_merge,
                              lg_pass_ir_->map_old_v_to_new_v_in_group_in);
              } else if (it.info.mode2 & TIMESTEP2_STORE) {
                auto storeOp_out = CreateStoreOp2(
                    it.value, it.info, ts_id, it.slice_idx, pipe_id,
                    lg_info.type, can_merge, l2mem_alloc_ptr);
                map_output_to_merge_slice[it.value].push_back(storeOp_out);
                map_group_out_to_yield_in[it.value] = storeOp_out;
              } else if (it.info.mode2 & TIMESTEP2_STORE_AND_LOAD) {
                if (ILP_time_step->mapILPVarInfo[it.varName].store_load_mode ==
                    0) {
                  auto storeOp_out = CreateStoreOp2(
                      it.value, it.info, ts_id, it.slice_idx, pipe_id,
                      lg_info.type, can_merge, l2mem_alloc_ptr);
                  if (std::find(lg_info.group_outs.begin(),
                                lg_info.group_outs.end(),
                                it.value) != lg_info.group_outs.end()) {
                    map_output_to_merge_slice[it.value].push_back(storeOp_out);
                    map_group_out_to_yield_in[it.value] = storeOp_out;
                  }
                } else {
                  CreateLoadOp2(ts_id, it, pipe_id, ops, ncdhw_idx, lg_info,
                                can_merge,
                                lg_pass_ir_->map_old_v_to_new_v_in_group_in);
                }
              }
            }
          }

          for (auto [op_idx, it] : llvm::enumerate(
                   ILP_time_step->timestep_table_new[ts].vec_op_infos)) {
            if (!it.op)
              continue;
            if (move_info.size() > 0 && op_idx > 0) {
              for (auto it3 : move_info) {
                if (it3.combine_ts_op_idx == op_idx) {
                  CreateLmemMoveOp(ts_id, it3);
                  ts_id++;
                }
              }
            }
            builder.setInsertionPointAfter(current_op_);
            auto new_op = builder.clone(*it.op);
            std::string tmpStr = "pipe" + std::to_string(pipe_id) + "_slice" +
                                 std::to_string(it.slice_idx);
            module::setLocSuffix(new_op, tmpStr);
            std::string name = module::getName(new_op).str();
            LOG(INFO) << "bdc_op_name: " << name;
            for (OpOperand &opd : it.op->getOpOperands()) {
              auto pre_op = opd.get().getDefiningOp();
              if (pre_op != nullptr && isa<top::NoneOp>(pre_op)) {
                continue;
              }
              Value new_value = opd.get();
              if (map_new_grp_out_to_old_grp_out.find(opd.get()) !=
                  map_new_grp_out_to_old_grp_out.end()) {
                // opd has been replaced with the output of the previous group
                new_value = map_old_to_new_value
                    [map_new_grp_out_to_old_grp_out[opd.get()]][it.slice_idx];
              } else {
                if (map_old_to_new_value.find(opd.get()) !=
                    map_old_to_new_value.end()) {
                  auto tmp_map = map_old_to_new_value[opd.get()];
                  if (tmp_map.find(it.slice_idx) != tmp_map.end()) {
                    new_value = tmp_map[it.slice_idx];
                  } else {
                    new_value = tmp_map[0]; // Use resident weights
                  }
                }
              }
              new_op->setOperand(opd.getOperandNumber(), new_value);
            }
            for (const auto &res : llvm::enumerate(it.op->getResults())) {
              map_old_to_new_value[res.value()][it.slice_idx] =
                  new_op->getResult(res.index());
            }
            auto ncdhw_idx =
                ILP_time_step->ncdhw_steps.begin()->second[it.slice_idx];
            UpdateOpLgParam2(new_op, it.op, ts_id, it.slice_idx, tensor_info,
                             ncdhw_idx, lg_info.type, can_merge);
            current_op_ = new_op;

            bool first_time = true;
            // In the case of composite op, the following processing should be
            // unified after the vec_op_infos traversal.
            //  The current processing inserts multiple dump time slots in the
            //  composite op
            // However, considering that dump is only a debugging function and
            // does not affect the normal running performance, the current
            // implementation is generally feasible, and no bugs have been
            // found, so do not modify it
            for (auto res : llvm::enumerate(it.op->getResults())) {
              if (std::find(will_store_value[pipe_id].begin(),
                            will_store_value[pipe_id].end(),
                            res.value()) == will_store_value[pipe_id].end()) {
                if (std::find(need_dump_tensor_values.begin(),
                              need_dump_tensor_values.end(),
                              res.value()) != need_dump_tensor_values.end()) {
                  if (first_time) {
                    ts_id++;
                    LOG(INFO) << "----------------ts" << ts_id
                              << " for op result dump-----------------";
                    first_time = false;
                  }
                  CreateStoreOp2(res.value(), tensor_info[res.value()], ts_id,
                                 it.slice_idx, pipe_id, lg_info.type, false);
                }
              }
            }
          }
        }
        if (!one_slice) {
          LOG(INFO) << "groupOp have many slice";
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
        LOG(INFO) << "groupOp.dump: ";
        groupOp.dump();
        current_op_ =
            groupOp; // The next group is inserted after the current group
      }
      if (!one_grp) {
        builder.setInsertionPointAfter(current_op_);
        llvm::SmallVector<NamedAttribute, 18> attrs;
        for (auto out : lg_info.group_outs) {
          auto tmp_op = builder.create<tpu::SliceMergeOp>(
              module::getLoc(out), out.getType(),
              map_output_to_merge_slice_for_grp[out], attrs);
          out.replaceUsesWithIf(tmp_op.getOutput(), [&](OpOperand &operand) {
            Operation *user = operand.getOwner();
            return find(ops.begin(), ops.end(), user) == ops.end();
          });
          map_old_grp_out_to_new_grp_out[out] = tmp_op.getOutput();
          map_new_grp_out_to_old_grp_out[tmp_op.getOutput()] = out;
          map_lg_extra_info[(int64_t)&lg_info].slice_merge_ops.push_back(
              tmp_op);
        }
      }
    }
  }

  if (print_node_topo_change) {
    idx = 0;
    for (auto node : topo_nodes) {
      if (node.global_op) {
        LOG(INFO) << "idx: " << idx
                  << " global_op: " << module::getName(node.global_op).str();
      } else {
        LOG(INFO) << "idx: " << idx << " group:";
        for (auto it : node.lgInfo->group_ops) {
          if (it)
            LOG(INFO) << "  local op: " << module::getName(it).str();
        }
      }
      idx++;
    }
  }

  Operation *firstOp = none_op_;
  for (auto node : topo_nodes) {
    if (node.global_op) {
      node.global_op->moveAfter(firstOp);
      if (print_node_topo_change) {
        if (isa<top::NoneOp>(firstOp))
          LOG(INFO) << "  move node, from: "
                    << module::getName(node.global_op).str() << ", to:NoneOp";
        else
          LOG(INFO) << "  move node, from: "
                    << module::getName(node.global_op).str()
                    << ", to:" << module::getName(firstOp).str();
      }
      firstOp = node.global_op;
    } else {
      for (auto it2 : map_lg_extra_info[(int64_t)node.lgInfo].buffer_ops) {
        it2->moveAfter(firstOp);
        if (print_node_topo_change) {
          if (isa<top::NoneOp>(firstOp))
            LOG(INFO) << "  move group, from: " << module::getName(it2).str()
                      << ", to:NoneOp";
          else
            LOG(INFO) << "  move group, from: " << module::getName(it2).str()
                      << ", to:" << module::getName(firstOp).str();
        }
        firstOp = it2;
      }
      for (auto it2 : map_lg_extra_info[(int64_t)node.lgInfo].outbuffer_ops) {
        it2->moveAfter(firstOp);
        if (print_node_topo_change) {
          if (isa<top::NoneOp>(firstOp))
            LOG(INFO) << "  move group, from: " << module::getName(it2).str()
                      << ", to:NoneOp";
          else
            LOG(INFO) << "  move group, from: " << module::getName(it2).str()
                      << ", to:" << module::getName(firstOp).str();
        }
        firstOp = it2;
      }
      for (auto it2 : map_lg_extra_info[(int64_t)node.lgInfo].grp_group_ops) {
        it2->moveAfter(firstOp);
        if (print_node_topo_change) {
          if (isa<top::NoneOp>(firstOp))
            LOG(INFO) << "  move group, from: " << module::getName(it2).str()
                      << ", to:NoneOp";
          else
            LOG(INFO) << "  move group, from: " << module::getName(it2).str()
                      << ", to:" << module::getName(firstOp).str();
        }
        firstOp = it2;
      }
      for (auto it2 : map_lg_extra_info[(int64_t)node.lgInfo].slice_merge_ops) {
        it2->moveAfter(firstOp);
        firstOp = it2;
      }
    }
  }
  retrunOp->moveBefore(&retrunOp->getBlock()->back());

  /*for (auto& node : nodes) {
    node.indeg = node.pre_nodes.size();
  }
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
