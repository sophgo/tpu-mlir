//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

class TruncIOWorker {
public:
  TruncIOWorker() {
    module = module::getAllModules()->at(0);
  }

  void add_input_name(std::string name) {
    input_names.insert(name);
  }

  void add_output_name(std::string name) {
    output_names.insert(name);
  }

  void invoke() {
    module.walk<WalkOrder::PreOrder>([&](func::FuncOp func) {
      if (func.getName().str() == "main") {
        return WalkResult::advance();
      }
      if (auto call = module::getCallOp(func)) {
        bool finish = call_subnet(call);
        if (finish)
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    module::removeUnusedOp();
  }

private:
  LayerGroupAttr get_lg_param(const LayerGroupAttr& ginfo, int64_t stage, int64_t id) {
    return LayerGroupAttr::get(module.getContext(),
                               ginfo.getOutAddr(),
                               ginfo.getOutSize(),
                               0, 0,
                               ginfo.getEuAlign(),
                               false,
                               ginfo.getInHsliceOffset(),
                               ginfo.getNIdx(),
                               ginfo.getNSlice(),
                               ginfo.getCIdx(),
                               ginfo.getCSlice(),
                               ginfo.getDIdx(),
                               ginfo.getDSlice(),
                               ginfo.getHIdx(),
                               ginfo.getHSlice(),
                               ginfo.getWIdx(),
                               ginfo.getWSlice(),
                               id,
                               stage,
                               0,
                               ginfo.getGroupType());
  }

  void set_LgOp_id(Operation* op, int64_t id) {
    LayerGroupAttr ginfo = op->getAttr("ginfo").cast<LayerGroupAttr>();
    auto param = LayerGroupAttr::get(module.getContext(),
                                    ginfo.getOutAddr(),
                                    ginfo.getOutSize(),
                                    ginfo.getBufferAddr(),
                                    ginfo.getBufferSize(),
                                    ginfo.getEuAlign(),
                                    ginfo.getCanMerge(),
                                    ginfo.getInHsliceOffset(),
                                    ginfo.getNIdx(),
                                    ginfo.getNSlice(),
                                    ginfo.getCIdx(),
                                    ginfo.getCSlice(),
                                    ginfo.getDIdx(),
                                    ginfo.getDSlice(),
                                    ginfo.getHIdx(),
                                    ginfo.getHSlice(),
                                    ginfo.getWIdx(),
                                    ginfo.getWSlice(),
                                    id,
                                    ginfo.getStage(),
                                    ginfo.getSliceIdx(),
                                    ginfo.getGroupType());
    op->setAttr(LocalGenInterface::kLayerGroupAttrName,
                param);
  }

  int64_t count_new_storeOps(Operation* op) {
    int64_t num = 0;
    for (auto v : op->getResults()) {
      if (module::isNone(v))
        continue;
      auto users = v.getUsers();
      bool stored = std::any_of(users.begin(), users.end(), [](auto user) {
        return isa<tpu::StoreOp>(user);
      });
      if (!stored)
        num++;
    }
    return num;
  }

  void try_insert_storeOps(Operation* op, llvm::SmallVector<Value>& new_groupOuts, int64_t start_id) {
    const uint64_t coeff_addr = module::getCoeffAddr(module);
    const uint64_t io_addr = module::getIOAddr(module);
    const uint64_t neuron_addr = module::getNeuronAddr(module);
    const uint64_t coeff_size = module::getCoeffSize(module);
    const uint64_t io_size = module::getIOSize(module);
    const uint64_t neuron_size = module::getNeuronSize(module);
    assert(coeff_addr + coeff_size <= neuron_addr);
    assert(io_addr + io_size <= neuron_addr);
    const int64_t start_addr = neuron_addr + neuron_size;
    int64_t running_start_addr = start_addr;
    auto ctx = module.getContext();
    OpBuilder builder(ctx);
    int64_t id = start_id;
    for (auto v : op->getResults()) {
      if (module::isNone(v))
        continue;
      bool stored = false;
      for (auto user : v.getUsers()) {
        if (auto storeOp = dyn_cast<tpu::StoreOp>(user)) {
          stored = true;
          new_groupOuts.push_back(storeOp.getOutput());
        }
      }
      if (stored)
        continue;
      std::vector<Value> operands;
      operands.push_back(v);
      operands.push_back(module::getNoneOp(op));
      std::string name = module::getName(v).str();
      std::vector<NamedAttribute> attrs;
      LayerGroupAttr ginfo = op->getAttr("ginfo").cast<LayerGroupAttr>();
      attrs.push_back(
        builder.getNamedAttr(
          LocalGenInterface::kLayerGroupAttrName,
          get_lg_param(ginfo, 2, id)
        )
      );
      id++;
      builder.setInsertionPointAfter(op);
      auto storeOp =
        builder.create<tpu::StoreOp>(NameLoc::get(builder.getStringAttr(name)),
                                      v.getType(), operands, attrs);
      auto store_out = storeOp.getOutput();
      module::setAddress(store_out, running_start_addr);
      running_start_addr += module::getBytes(store_out);
      new_groupOuts.push_back(store_out);
    }
    module::setNeuronSize(module, running_start_addr - neuron_addr);
  }

  int64_t get_group_ts_rows(tpu::GroupOp groupOp, std::vector<std::vector<int64_t>>& ts_rows) {
    auto flow = module::getI64Array(groupOp.getFlow());
    int64_t max_id = 0;
    std::vector<int64_t> ts_row;
    for (size_t i = 1; i < flow->size(); ++i) {
      if (flow->at(i) < 0) {
        ts_rows.push_back(ts_row);
        ts_row.clear();
        continue;
      }
      ts_row.push_back(flow->at(i));
      max_id = std::max(max_id, flow->at(i));
    }
    ts_rows.push_back(ts_row);
    return max_id;
  }

  void set_group_ts_rows(tpu::GroupOp groupOp, const std::vector<std::vector<int64_t>>& ts_rows) {
    auto ctx = module.getContext();
    OpBuilder builder(ctx);
    std::vector<int64_t> flow;
    for (auto i = 0; i < ts_rows.size(); ++i) {
      flow.push_back(-i);
      flow.insert(flow.end(), ts_rows[i].begin(), ts_rows[i].end());
    }
    groupOp->setAttr("flow", builder.getI64ArrayAttr(flow));
  }

  llvm::SmallVector<int64_t> analyze_group_data_flow(tpu::GroupOp groupOp, Operation* end_op) {
    llvm::SmallDenseMap<Operation*, int64_t> op_ids;
    auto &body = groupOp.getBody().front();
    auto& ops = body.getOperations();
    for (auto& op : ops) {
      if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
        auto gi = lgOp.getGroupInfo((int64_t)0, (int64_t)0, (int64_t)0, (int64_t)0, (int64_t)0);
        op_ids[&op] = gi.id;
      }
    }
    auto end_op_id = op_ids[end_op];
    llvm::SmallDenseMap<int64_t, llvm::SmallDenseSet<int64_t>> defop_ids;
    for (auto& op : ops) {
      if (auto lgOp = dyn_cast<LocalGenInterface>(op)) {
        auto op_id = op_ids[&op];
        defop_ids[op_id] = llvm::SmallDenseSet<int64_t>();
        for (auto v : op.getOperands()) {
          auto def_op = v.getDefiningOp();
          auto defop_id = def_op ? op_ids[def_op] : -1;
          defop_ids[op_id].insert(defop_id);
        }
      }
    }
    llvm::SetVector<int64_t> live_ids;
    live_ids.insert(end_op_id);
    for (int i = 0; i < live_ids.size(); ++i) {
      for (auto id : defop_ids[live_ids[i]]) {
        if (id != -1) {
          live_ids.insert(id);
        }
      }
    }
    auto vec = live_ids.takeVector();
    llvm::sort(vec.begin(), vec.end());
    return vec;
  }

  void fix_group_flow(tpu::GroupOp groupOp, const llvm::SmallVector<int64_t>& live_ids, int64_t num_new_st) {
    std::vector<std::vector<int64_t>> ts_rows;
    get_group_ts_rows(groupOp, ts_rows);
    auto &body = groupOp.getBody().front();
    auto& ops = body.getOperations();
    llvm::SmallDenseMap<int64_t, int64_t> id_map;
    int64_t new_id = 0;
    for (auto id : live_ids) {
      id_map[id] = new_id;
      new_id++;
    }
    for (auto [old_id, op] : llvm::enumerate(ops)) {
      if (id_map.count(old_id)) {
        set_LgOp_id(&op, id_map[old_id]);
      }
    }
    for (auto i = 0; i < ts_rows.size(); ++i) {
      for (auto j = 0; j < ts_rows[i].size(); ++j) {
        auto& id = ts_rows[i][j];
        if (!id_map.count(id)) {
          ts_rows[i].erase(ts_rows[i].begin() + j);
          j--;
        } else {
          id = id_map[id];
        }
      }
      if (ts_rows[i].empty()) {
        ts_rows.erase(ts_rows.begin() + i);
        i--;
      }
    }
    std::vector<int64_t> st_ts_row;
    for (int i = new_id; i < new_id + num_new_st; ++i) {
      st_ts_row.push_back(i);
    }
    ts_rows.emplace_back(std::move(st_ts_row));
    set_group_ts_rows(groupOp, ts_rows);
  }

  bool is_input(mlir::Value v) {
    if (module::isNone(v))
      return false;
    auto name = module::getName(v).str();
    return input_names.find(name) != input_names.end();
  }

  bool is_output(mlir::Value v) {
    if (module::isNone(v))
      return false;
    auto name = module::getName(v).str();
    return output_names.find(name) != output_names.end();
  }

  bool call_subnet(func::CallOp callOp) {
    auto funcOp = module::getFuncOp(module, callOp.getCallee());
    auto retOp = funcOp.getBody().back().getTerminator();
    auto ctx = module.getContext();
    OpBuilder builder(ctx);
    llvm::SmallVector<mlir::Value> new_outs;
    bool output_found = false;
    Operation *op_to_check = nullptr;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto groupOp = dyn_cast<GroupOp>(op)) {
        auto &body = groupOp.getBody().front();
        auto yieldOp = body.getTerminator();
        llvm::SmallVector<Value> new_groupOuts;
        body.walk([&](Operation *lop) {
          if (auto lgOp = dyn_cast<LocalGenInterface>(lop)) {
            for (auto v : lop->getResults()) {
              if (is_output(v)) {
                op_to_check = lop;
                output_found = true;
                break;
              }
            }
            if (output_found) {
              assert(lop->getNumResults() == 1);
              auto live_ids = analyze_group_data_flow(groupOp, lop);
              auto num_new_st = count_new_storeOps(lop);
              fix_group_flow(groupOp, live_ids, num_new_st);
              try_insert_storeOps(lop, new_groupOuts, live_ids.size());
              yieldOp->setOperands(new_groupOuts);
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (output_found) {
          llvm::SmallVector<Type, 8> ret_types;
          llvm::SmallVector<Location, 8> locs;
          for (auto out : new_groupOuts) {
            ret_types.push_back(out.getType());
            locs.push_back(module::getLoc(out));
          }
          auto group_loc = builder.getFusedLoc(locs);
          builder.setInsertionPoint(groupOp);
          auto new_groupOp =
            builder.create<tpu::GroupOp>(
              group_loc, ret_types, groupOp->getOperands(), groupOp->getAttrs());
          new_groupOp.getBody().takeBody(groupOp.getBody());
          for (auto v : new_groupOp->getResults()) {
            new_outs.push_back(v);
          }
          // [trick] fill block of groupOp with a terminator
          groupOp.getBody().emplaceBlock();
          builder.setInsertionPointToStart(&(groupOp.getBody().front()));
          builder.create<tpu::YieldOp>(groupOp->getLoc(), groupOp->getOperands());
          return WalkResult::interrupt();
        }
        return WalkResult::skip();
      }

      if (auto globalOp = dyn_cast<GlobalGenInterface>(op)) {
        auto op_res = op->getResults();
        for (auto v : op_res) {
          if (is_output(v)) {
            op_to_check = op;
            output_found = true;
            break;
          }
        }
        if (output_found) {
          for (auto v : op_res) {
            new_outs.push_back(v);
          }
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (output_found) {
      retOp->setOperands(new_outs);
      llvm::SmallVector<mlir::Type> new_outTypes;
      for (auto v: new_outs) {
        new_outTypes.push_back(v.getType());
      }
      funcOp.setType(
        mlir::FunctionType::get(
          funcOp.getContext(),
          funcOp.getArgumentTypes(),
          new_outTypes));
      auto mainFuncOp = module::getMainFuncOp(module);
      auto mainRetOp = mainFuncOp.getBody().back().getTerminator();
      builder.setInsertionPoint(mainRetOp);
      auto new_callOp = builder.create<func::CallOp>(
        builder.getUnknownLoc(), funcOp, callOp.getArgOperands());
      std::set<Operation*> ops_to_del;
      llvm::SmallVector<mlir::Value> old_mainOuts;
      for (auto opd : mainRetOp->getOperands()) {
        auto def_op = opd.getDefiningOp();
        if (ops_to_del.find(def_op) == ops_to_del.end()) {
          ops_to_del.insert(def_op);
          old_mainOuts.push_back(opd);
        }
      }
      mainRetOp->setOperands(new_callOp->getResults());
      for (auto opd : old_mainOuts) {
        auto def_op = opd.getDefiningOp();
        def_op->dropAllReferences();
        def_op->dropAllDefinedValueUses();
        def_op->erase();
      }
      mainFuncOp.setType(
        mlir::FunctionType::get(
          mainFuncOp.getContext(),
          mainFuncOp.getArgumentTypes(),
          new_outTypes));
    }
    return output_found;
  }

  ModuleOp module;
  std::set<std::string> input_names;
  std::set<std::string> output_names;
};

class TruncIOPass : public TruncIOBase<TruncIOPass> {
public:
  TruncIOPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    module::init(mOp);
    if (!module::isState(module::State::TPU_ADDRESSED)) {
      llvm_unreachable("module should be addressed");
    }
    TruncIOWorker worker;
    std::stringstream ss;
    std::string item;
    // ss.str(inputs);
    // while (std::getline(ss, item, ',')) {
    //   worker.add_input_name(item);
    // }
    // ss.clear();
    ss.str(outputs);
    while (std::getline(ss, item, ',')) {
      worker.add_output_name(item);
    }
    worker.invoke();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createTruncIOPass() {
  return std::make_unique<TruncIOPass>();
}
} // namespace tpu
} // namespace tpu_mlir
