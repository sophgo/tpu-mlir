//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/LayerGroup/LayerGroupDefs.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <llvm/ADT/PointerEmbeddedInt.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace tpu_mlir {
namespace tpu {

using mlir::Type;
using mlir::Value;

namespace moduleTool {

std::vector<Value> collectValuesByName(ModuleOp submodule,
                                       const std::string &name,
                                       bool remove_local_op = false) {
  std::vector<Value> results;
  for (auto func : submodule.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<func::CallOp>(op))
        return WalkResult::skip();
      for (auto out : op->getResults()) {
        if (module::getName(out) == name) {
          results.push_back(out);
        }
      }
      return WalkResult::advance();
    });
  }
  assert(results.size() > 0 && "Value not found");
  // sort values w.r.t. inputOp > globalOp > localOp.
  std::sort(results.begin(), results.end(), [&](Value a, Value b) {
    auto getOpPriority = [](Operation *op) -> int {
      return isa<top::InputOp>(op)
                 ? 0
                 : !module::isOpInGroup(op) ? 1 : isa<tpu::StoreOp>(op) ? 2 : 3;
    };
    return getOpPriority(a.getDefiningOp()) < getOpPriority(b.getDefiningOp());
  });
  // remove local op for addr assignment uses. (storeop etc. are kept.)
  if (remove_local_op) {
    llvm::erase_if(results, [](Value v) {
      return module::isOpInGroup(v.getDefiningOp()) &&
             !isa<tpu::StoreOp, tpu::YieldOp>(v.getDefiningOp());
    });
  }
  return results;
}

int64_t assign_new_addrs(std::vector<Value> &values, int64_t addr_limit) {
  const int64_t alignment = backend::BM168x::ALIGNMENT;
  auto bytes = module::getBytes(values[0]);
  const int64_t new_addr = align_up(addr_limit, alignment);
  for (auto value : values) {
    module::setAddress(value, new_addr);
  }
  addr_limit = new_addr + align_up(bytes, alignment);
  return addr_limit;
}

std::vector<llvm::StringRef> get_io_names(ModuleOp submodule) {
  std::vector<Value> io_values;
  module::getInputsOutputs(submodule, io_values, io_values);
  std::vector<llvm::StringRef> io_names;
  for (auto value : io_values) {
    io_names.emplace_back(module::getName(value));
  }
  return io_names;
}
} // namespace moduleTool

namespace groupTool {

static LayerGroupAttr create_lg_param_like(Value local_value, int64_t new_stage,
                                           int64_t new_id) {
  LayerGroupAttr ginfo =
      local_value.getDefiningOp()->getAttr("ginfo").cast<LayerGroupAttr>();
  auto ctx = local_value.getDefiningOp()->getContext();
  return LayerGroupAttr::get(
      ctx, ginfo.getOutAddr(), ginfo.getOutSize(), 0, 0, ginfo.getEuAlign(),
      false, ginfo.getInHsliceOffset(), ginfo.getNIdx(), ginfo.getNSlice(),
      ginfo.getCIdx(), ginfo.getCSlice(), ginfo.getDIdx(), ginfo.getDSlice(),
      ginfo.getHIdx(), ginfo.getHSlice(), ginfo.getWIdx(), ginfo.getWSlice(),
      new_id, new_stage, 0, ginfo.getGroupType());
}

static int64_t
parse_timestep_table(tpu::GroupOp gOp,
                     std::vector<std::vector<int64_t>> &timestep_table) {
  timestep_table.clear();
  auto flow = module::getI64Array(gOp.getFlow());
  std::vector<int64_t> ts_row;
  int64_t max_id = 0;
  for (size_t i = 1; i < flow->size(); ++i) {
    if (flow->at(i) < 0) {
      timestep_table.push_back(ts_row);
      ts_row.clear();
      continue;
    }
    ts_row.push_back(flow->at(i));
    max_id = std::max(max_id, flow->at(i));
  }
  timestep_table.push_back(ts_row);
  return max_id;
}

void create_load_op(Value load_value) {
  // 1. input of new load_op should be a func arg.
  auto func_op = load_value.getDefiningOp()->getParentOfType<FuncOp>();
  Value func_arg = nullptr;
  for (auto arg : func_op.getArguments()) {
    if (module::getName(arg) == module::getName(load_value)) {
      func_arg = arg;
      break;
    }
  }
  assert(func_arg && "func_ar not found");
  // 2. re-construct the groupOp with one more input.
  auto gOp = cast<tpu::GroupOp>(load_value.getDefiningOp()->getParentOp());
  OpBuilder builder(gOp);
  std::vector<Type> result_types;
  for (auto res : gOp.getResults()) {
    result_types.push_back(res.getType());
  }
  std::vector<Value> new_inputs;
  for (auto input : gOp.getInputs()) {
    new_inputs.push_back(input);
  }
  new_inputs.push_back(func_arg);
  auto new_gOp = builder.create<tpu::GroupOp>(gOp.getLoc(), result_types,
                                              new_inputs, gOp->getAttrs());
  for (auto it : llvm::enumerate(gOp.getResults())) {
    it.value().replaceAllUsesWith(new_gOp.getResults()[it.index()]);
  }
  new_gOp.getBody().takeBody(gOp.getBody());

  // 3. create new load_op and replace the old tpu::local_op.
  builder.setInsertionPointAfter(load_value.getDefiningOp());
  auto ginfo =
      load_value.getDefiningOp()->getAttr("ginfo").cast<LayerGroupAttr>();
  std::vector<NamedAttribute> load_attrs;
  load_attrs.push_back(builder.getNamedAttr(
      "lmem_type", builder.getI64IntegerAttr(LMEM_ACTIVATION)));
  load_attrs.push_back(builder.getNamedAttr(
      LocalGenInterface::kLayerGroupAttrName,
      create_lg_param_like(load_value, ginfo.getStage(), ginfo.getId())));
  auto load_op = builder.create<tpu::LoadOp>(
      NameLoc::get(
          builder.getStringAttr("load_" + module::getName(load_value).str())),
      load_value.getType(), func_arg, load_attrs);
  load_value.replaceAllUsesWith(load_op.getOutput());
  gOp->erase();

  // 4. fix timestep table. assign new loadop with unique timestep.
  std::vector<std::vector<int64_t>> timestep_table, new_timestep_table;
  parse_timestep_table(new_gOp, timestep_table);
  int64_t load_id = ginfo.getId();
  for (auto &ts_row : timestep_table) {
    if (std::find(ts_row.begin(), ts_row.end(), load_id) == ts_row.end()) {
      new_timestep_table.push_back(ts_row);
    } else {
      for (auto id : ts_row) {
        new_timestep_table.push_back({id});
      }
    }
  }
  auto flow = std::vector<int64_t>();
  for (auto ts_row : llvm::enumerate(new_timestep_table)) {
    flow.push_back(-(ts_row.index() + 1));
    flow.insert(flow.end(), ts_row.value().begin(), ts_row.value().end());
  }
  new_gOp->setAttr("flow", builder.getI64ArrayAttr(flow));
}

Value create_store_op(Value store_value, bool store_after_producer) {
  assert(module::isOpInGroup(store_value.getDefiningOp()));
  auto gOp = cast<tpu::GroupOp>(store_value.getDefiningOp()->getParentOp());
  OpBuilder builder(gOp);
  std::vector<std::vector<int64_t>> timestep_table, new_timestep_table;
  int64_t max_id = parse_timestep_table(gOp, timestep_table);
  max_id++;          // id for new store.
  int64_t stage = 0; // stage for new store.

  // 1. re-construct data flow.
  if (store_after_producer) {
    /** put storeOp after its producer. */
    LayerGroupAttr producer_ginfo =
        store_value.getDefiningOp()->getAttr("ginfo").cast<LayerGroupAttr>();
    stage = producer_ginfo.getStage();
    int64_t producer_id = producer_ginfo.getId();
    for (auto &ts_row : timestep_table) {
      if (std::find(ts_row.begin(), ts_row.end(), producer_id) ==
          ts_row.end()) {
        new_timestep_table.push_back(ts_row);
      } else {
        for (auto id : ts_row) {
          new_timestep_table.push_back({id});
          if (id == producer_id)
            new_timestep_table.push_back({max_id});
        }
      }
    }
  } else {
    /** put storeOp before its last consumer. */
    Operation *last_user;
    {
      std::vector<Operation *> users;
      for (auto user : store_value.getUsers())
        users.push_back(user);
      std::sort(users.begin(), users.end(), [&](Operation *a, Operation *b) {
        auto a_ginfo = a->getAttr(LocalGenInterface::kLayerGroupAttrName)
                           .cast<LayerGroupAttr>();
        auto b_ginfo = b->getAttr(LocalGenInterface::kLayerGroupAttrName)
                           .cast<LayerGroupAttr>();
        return a_ginfo.getStage() < b_ginfo.getStage() ||
               (a_ginfo.getStage() == b_ginfo.getStage() &&
                a_ginfo.getId() < b_ginfo.getId());
      });
      last_user = users.back();
    }
    LayerGroupAttr last_ginfo =
        last_user->getAttr("ginfo").cast<LayerGroupAttr>();
    stage = last_ginfo.getStage();
    int64_t last_id = last_ginfo.getId();
    for (auto &ts_row : timestep_table) {
      if (std::find(ts_row.begin(), ts_row.end(), last_id) == ts_row.end()) {
        new_timestep_table.push_back(ts_row);
      } else {
        for (auto id : ts_row) {
          if (id == last_id)
            new_timestep_table.push_back({max_id});
          new_timestep_table.push_back({id});
        }
      }
    }
  }
  auto flow = std::vector<int64_t>();
  for (auto ts_row : llvm::enumerate(new_timestep_table)) {
    flow.push_back(-(ts_row.index() + 1));
    flow.insert(flow.end(), ts_row.value().begin(), ts_row.value().end());
  }
  gOp->setAttr("flow", builder.getI64ArrayAttr(flow));

  // 2. create new storeOp, update tpu.YieldOp.
  auto &block = gOp.getBody().front();
  Operation *yield_op = block.getTerminator();
  std::vector<NamedAttribute> store_attrs;
  store_attrs.push_back(
      builder.getNamedAttr(LocalGenInterface::kLayerGroupAttrName,
                           create_lg_param_like(store_value, stage, max_id)));
  builder.setInsertionPoint(yield_op);
  std::vector<Value> operands;
  operands.push_back(store_value);
  operands.push_back(module::getNoneOp(store_value.getDefiningOp()));
  auto store_op = builder.create<tpu::StoreOp>(
      store_value.getLoc(), store_value.getType(), operands, store_attrs);

  // create new yieldOp
  llvm::SmallVector<Location, 8> locs;
  llvm::SmallVector<Value, 8> store_values;
  for (auto out : yield_op->getOperands()) {
    locs.push_back(module::getLoc(out));
    store_values.push_back(out);
  }
  locs.push_back(module::getLoc(store_op));
  store_values.push_back(store_op.getOutput());
  auto group_loc = builder.getFusedLoc(locs);
  builder.create<tpu::YieldOp>(group_loc, store_values);
  yield_op->erase();

  // 3. re-construct groupOp.
  builder.setInsertionPointAfter(gOp);
  std::vector<Type> new_result_types;
  for (auto res : gOp.getResults()) {
    new_result_types.push_back(res.getType());
  }
  new_result_types.push_back(store_value.getType());
  auto new_gOp = builder.create<tpu::GroupOp>(group_loc, new_result_types,
                                              gOp.getInputs(), gOp->getAttrs());
  new_gOp.getBody().takeBody(gOp.getBody());
  for (auto it : llvm::enumerate(gOp.getResults())) {
    it.value().replaceAllUsesWith(new_gOp.getResults()[it.index()]);
  }
  gOp->erase();
  return new_gOp.getOutputs().back();
}

tpu::GroupOp prune_group_op_outputs(tpu::GroupOp op) {
  /** remove unused group-outputs. */
  std::vector<int> kept_out_ids;
  for (auto it : llvm::enumerate(op->getResults())) {
    if (it.value().use_empty()) {
      continue;
    }
    kept_out_ids.push_back(it.index());
  }
  if (kept_out_ids.size() == op->getResults().size()) {
    return op;
  }
  OpBuilder builder(op->getContext());
  llvm::SmallVector<Location, 8> locs;
  for (auto kept_out_id : kept_out_ids) {
    locs.push_back(module::getLoc(op->getResults()[kept_out_id]));
  }
  auto new_loc = builder.getFusedLoc(locs);

  // re-construct groupOp.
  llvm::SmallVector<Type, 8> new_result_types;
  for (auto kept_out_id : kept_out_ids) {
    new_result_types.push_back(op->getResults()[kept_out_id].getType());
  }
  builder.setInsertionPoint(op);
  auto new_gOp = builder.create<tpu::GroupOp>(new_loc, new_result_types,
                                              op.getInputs(), op->getAttrs());
  new_gOp.getBody().takeBody(op.getBody());

  // create new yieldOp.
  auto yield_op = new_gOp.getBody().front().getTerminator();
  llvm::SmallVector<Value, 8> new_store_values;
  for (auto kept_out_id : kept_out_ids) {
    new_store_values.push_back(yield_op->getOperand(kept_out_id));
  }
  builder.setInsertionPoint(yield_op);
  builder.create<tpu::YieldOp>(new_loc, new_store_values);
  yield_op->erase();

  for (auto kept_out_id : llvm::enumerate(kept_out_ids)) {
    op->getResult(kept_out_id.value())
        .replaceAllUsesWith(new_gOp.getResult(kept_out_id.index()));
  }
  op->erase();
  return new_gOp;
}

void prune_group_op_inputs(tpu::GroupOp op) {
  /** remove unused group-inputs. */
  std::vector<int> kept_in_ids;
  for (auto it : llvm::enumerate(op.getInputs())) {
    op.walk([&](tpu::LoadOp load_op) {
      if (load_op.getInput() == it.value()) {
        kept_in_ids.push_back(it.index());
      }
    });
  }
  // re-construct groupOp.
  OpBuilder builder(op);
  llvm::SmallVector<Value> new_inputs;
  for (auto kept_in_id : kept_in_ids) {
    new_inputs.push_back(op.getInputs()[kept_in_id]);
  }
  llvm::SmallVector<Type> result_types;
  for (auto res : op.getResults()) {
    result_types.push_back(res.getType());
  }
  auto new_gOp = builder.create<tpu::GroupOp>(op.getLoc(), result_types,
                                              new_inputs, op->getAttrs());
  for (auto it : llvm::enumerate(op.getResults())) {
    it.value().replaceAllUsesWith(new_gOp.getResult(it.index()));
  }
  new_gOp.getBody().takeBody(op.getBody());
  op->erase();
}
} // namespace groupTool

namespace funcTool {

static void update_func_rets(FuncOp func, llvm::SmallVector<Value> &rets) {
  llvm::SmallVector<Type> new_ret_types;
  for (auto ret : rets) {
    new_ret_types.push_back(ret.getType());
  }
  func.setType(mlir::FunctionType::get(func.getContext(),
                                       func.getArgumentTypes(), new_ret_types));
  func.walk([&](func::ReturnOp returnOp) {
    OpBuilder builder(returnOp);
    builder.create<func::ReturnOp>(returnOp.getLoc(), rets);
    returnOp.erase();
  });
}

static Value create_new_arg(FuncOp &funcOp, OpBuilder &builder, Value v,
                            bool is_main_func = false) {
  auto &block = funcOp.getFunctionBody().front();
  auto arg = is_main_func
                 ? block.addArgument(v.getType(), builder.getUnknownLoc())
                 : block.addArgument(v.getType(), v.getLoc());
  funcOp.setType(mlir::FunctionType::get(
      funcOp.getContext(), block.getArgumentTypes(), funcOp.getResultTypes()));
  return arg;
}

Value create_input_op(Value ori_value, ModuleOp submodule) {
  auto main_func = module::getMainFuncOp(submodule);
  auto builder = OpBuilder::atBlockBegin(&main_func.front());
  const auto &arg = create_new_arg(main_func, builder, ori_value, true);
  auto input_op = builder.create<top::InputOp>(
      ori_value.getLoc(), ori_value.getType(), ValueRange{arg});
  return input_op.getOutput();
}

void get_func_args_and_res(FuncOp func_op, std::vector<Value> &fnargs,
                           std::vector<Value> &fnres) {
  fnargs.clear();
  fnres.clear();
  for (auto arg : func_op.getArguments()) {
    fnargs.push_back(arg);
  }
  func_op.walk([&](ReturnOp op) {
    for (auto output : op.getOperands()) {
      fnres.push_back(output);
    }
  });
}

void get_fake_returns(FuncOp func, std::vector<int> &fake_args_ids,
                      std::vector<int> &fake_rets_ids) {
  func.walk([&](func::ReturnOp returnOp) {
    for (auto it : llvm::enumerate(returnOp.getOperands())) {
      auto output = it.value();
      int ret_id = it.index();
      if (auto blockArg = dyn_cast<BlockArgument>(output)) {
        fake_args_ids.push_back(blockArg.getArgNumber());
        fake_rets_ids.push_back(ret_id);
      }
    }
  });
}

void augment_sub_funcs_with_new_args(std::vector<Value> &new_inputs,
                                     ModuleOp submodule) {
  auto argument_exists = [](FuncOp func, Value value) {
    auto name = module::getName(value);
    return std::any_of(func.getArguments().begin(), func.getArguments().end(),
                       [&](Value arg) { return module::getName(arg) == name; });
  };
  for (auto func : submodule.getOps<FuncOp>()) {
    if (func.getName() == "main" || argument_exists(func, new_inputs[0])) {
      continue;
    }
    auto builder = OpBuilder::atBlockBegin(&func.front());
    auto new_arg = create_new_arg(func, builder, new_inputs[0]);
    for (auto new_input : new_inputs) {
      if (new_input.getDefiningOp()->getParentRegion() !=
          &func.getFunctionBody()) {
        continue;
      }
      new_input.replaceAllUsesWith(new_arg);
    }
  }
}

void augment_sub_funcs_with_new_rets(std::vector<Value> &new_outputs) {
  for (auto new_output : new_outputs) {
    if (module::isOpInGroup(new_output.getDefiningOp()) ||
        isa<top::InputOp, tpu::YieldOp, tpu::StoreOp, tpu::CoreJoinOp>(
            new_output.getDefiningOp())) {
      continue;
    }
    auto func = new_output.getDefiningOp()->getParentOfType<FuncOp>();
    bool value_returned = false;
    llvm::SmallVector<Value> rets;
    func.walk([&](func::ReturnOp returnOp) {
      for (auto output : returnOp.getOperands()) {
        if (output == new_output) {
          value_returned = true;
        }
        rets.push_back(output);
      }
    });
    if (!value_returned) {
      rets.push_back(new_output);
      update_func_rets(func, rets);
    }
  }
}

void augment_main_func_with_new_ios(
    ModuleOp submodule, std::vector<std::string> &expected_output_names) {
  auto main_func = module::getMainFuncOp(submodule);
  OpBuilder builder(main_func);

  std::map<std::string, mlir::Value> live_values;
  main_func.walk([&](top::InputOp input_op) {
    auto input_op_name = module::getName(input_op, 0).str();
    live_values[input_op_name] = input_op.getOutput();
  });
  // 1. update callOp forwardly.
  main_func.walk([&](func::CallOp call_op) {
    auto func_op = module::getFuncOp(submodule, call_op.getCallee());
    std::vector<Value> fnArgs, fnRes, newInputs;
    llvm::SmallVector<Type, 8> newResTypes;
    get_func_args_and_res(func_op, fnArgs, fnRes);
    for (auto arg : fnArgs) {
      newInputs.push_back(live_values[module::getName(arg).str()]);
    }
    for (auto res : fnRes) {
      newResTypes.push_back(res.getType());
    }
    builder.setInsertionPointAfter(call_op);
    // create new call.
    auto new_call_op = builder.create<func::CallOp>(
        call_op.getLoc(), func_op.getName(), newResTypes, newInputs);
    for (auto it : llvm::enumerate(call_op.getResults())) {
      it.value().replaceAllUsesWith(new_call_op.getResult(it.index()));
    }
    call_op.erase();
    for (auto it : llvm::enumerate(new_call_op.getResults())) {
      live_values[module::getName(fnRes[it.index()]).str()] = it.value();
    }
  });
  // 2. update rets.
  SmallVector<Value> newReturns;
  for (const auto &name : expected_output_names) {
    if (live_values.find(name) != live_values.end()) {
      newReturns.push_back(live_values[name]);
    }
  }
  update_func_rets(main_func, newReturns);
}

void prune_func_args(FuncOp func) {
  auto &block = func.getFunctionBody().front();
  for (int i = block.getNumArguments() - 1; i >= 0; --i) {
    auto arg = block.getArgument(i);
    if (arg.use_empty()) {
      block.eraseArgument(i);
    }
  }
  func.setType(mlir::FunctionType::get(
      func.getContext(), block.getArgumentTypes(), func.getResultTypes()));
}

static bool prune_func_body(FuncOp func, bool remove_local_op) {
  bool any_op_removed = false;
  std::vector<Operation *> all_ops;
  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<ReturnOp, FuncOp, tpu::YieldOp, top::InputOp>(op))
      all_ops.push_back(op);
  });
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if (!remove_local_op && module::isOpInGroup((*iter))) {
      continue;
    }
    if ((*iter)->use_empty()) {
      (*iter)->erase();
      any_op_removed = true;
    }
  }
  return any_op_removed;
}

static void prune_func(FuncOp func, std::vector<int> &kept_in_ids,
                       const std::vector<int> &kept_out_ids,
                       const bool remove_unused_local_ops) {
  // prune unused rets.
  llvm::SmallVector<Value> new_rets;
  func.walk([&](func::ReturnOp returnOp) {
    for (auto kept_out_id : kept_out_ids) {
      new_rets.push_back(returnOp.getOperand(kept_out_id));
    }
  });
  update_func_rets(func, new_rets);
  do {
    if (remove_unused_local_ops) {
      func.walk([&](tpu::GroupOp op) {
        auto new_gOp = groupTool::prune_group_op_outputs(op);
        groupTool::prune_group_op_inputs(new_gOp);
      });
    }
  } while (prune_func_body(func, remove_unused_local_ops));
  // prune unused args.
  for (auto arg : func.getArguments()) {
    if (!arg.use_empty()) {
      kept_in_ids.push_back(arg.getArgNumber());
    }
  }
  prune_func_args(func);
}

void prune_unused_vars(ModuleOp submodule, const bool remove_unused_local_ops) {

  std::set<std::string> used_values;
  auto main_func = module::getMainFuncOp(submodule);
  { // push main-func-outs to used_values.
    std::vector<Value> inputs, outputs;
    module::getInputsOutputs(submodule, inputs, outputs);
    for (auto &output : outputs) {
      used_values.insert(module::getName(output).str());
    }
  }
  // 1. Reverse order & Recursively prune.
  // mainfunc->callOp->subfunc->groupOp->...
  OpBuilder builder(main_func);
  std::vector<func::CallOp> call_ops;
  main_func.walk([&](func::CallOp call_op) { call_ops.push_back(call_op); });
  // remove fake args and rets, where ret = arg.
  for (auto it = call_ops.rbegin(); it != call_ops.rend(); ++it) {
    auto sub_func = module::getFuncOp(submodule, it->getCallee());
    std::vector<int> fake_args_ids, fake_rets_ids;
    get_fake_returns(sub_func, fake_args_ids, fake_rets_ids);
    for (int i = 0; i < fake_args_ids.size(); ++i) {
      it->getResult(fake_rets_ids[i])
          .replaceAllUsesWith(it->getOperand(fake_args_ids[i]));
    }
  }
  for (auto it = call_ops.rbegin(); it != call_ops.rend(); ++it) {
    func::CallOp call_op = *it;
    auto sub_func = module::getFuncOp(submodule, call_op.getCallee());
    std::vector<Value> fnArgs, fnRes;
    get_func_args_and_res(sub_func, fnArgs, fnRes);
    std::vector<int> kept_in_ids, kept_out_ids;
    // 1.1. prune func outs.
    for (auto it : llvm::enumerate(fnRes)) {
      if (used_values.find(module::getName(it.value()).str()) !=
          used_values.end()) {
        if (std::find(fnArgs.begin(), fnArgs.end(), it.value()) !=
            fnArgs.end()) {
          continue; // fake rets.
        }
        kept_out_ids.push_back(it.index());
      }
    }
    if (kept_out_ids.size() == 0) {
      sub_func.erase();
      call_op.erase();
      continue;
    }
    // 1.2. prune func body.
    prune_func(sub_func, kept_in_ids, kept_out_ids, remove_unused_local_ops);
    get_func_args_and_res(sub_func, fnArgs, fnRes);
    // 1.3. prune func args.
    builder.setInsertionPoint(call_op);
    llvm::SmallVector<Type> new_fout_types;
    llvm::SmallVector<Value> new_fins;
    for (auto in_id : kept_in_ids) {
      new_fins.push_back(call_op.getOperand(in_id));
    }
    for (auto fn_res : fnRes) {
      new_fout_types.push_back(fn_res.getType());
    }
    auto new_call_op = builder.create<func::CallOp>(
        call_op.getLoc(), sub_func.getName(), new_fout_types, new_fins);
    for (auto it : llvm::enumerate(new_call_op.getResults())) {
      call_op.getResult(kept_out_ids[it.index()])
          .replaceAllUsesWith(it.value());
    }
    call_op.erase();
    for (auto fn_arg : fnArgs) {
      used_values.insert(module::getName(fn_arg).str());
    }
  }
  // prune unused inputOp in main_func.
  main_func.walk([&](top::InputOp input_op) {
    if (used_values.find(module::getName(input_op, 0).str()) ==
        used_values.end()) {
      input_op.erase();
    }
  });
  // update main_func args.
  prune_func_args(main_func);

  // 2.1. update groupOp params: (1) flow, (2) other_up_overlap_op, (3)
  // other_down_overlap_op.
  submodule.walk([&](tpu::GroupOp op) {
    auto flow = module::getI64Array(op.getFlow());
    // update flow. remove empty timestep. remove dead ids.
    std::set<int64_t> lived_ids;
    op.getBody().walk([&](Operation *op) {
      if (op->hasAttr("ginfo")) {
        auto ginfo = op->getAttr("ginfo").cast<LayerGroupAttr>();
        lived_ids.insert(ginfo.getId());
      }
    });
    std::vector<int64_t> new_flow;
    int64_t ts = -1;
    for (auto id : *flow) {
      if (id < 0 && (new_flow.empty() || new_flow.back() >= 0)) {
        new_flow.push_back(ts--);
        continue;
      }
      if (lived_ids.find(id) != lived_ids.end()) {
        new_flow.push_back(id);
      }
    }
    op->setAttr("flow", builder.getI64ArrayAttr(new_flow));
    op->setAttr("other_up_overlap_op", builder.getI64ArrayAttr({}));
    op->setAttr("other_down_overlap_op", builder.getI64ArrayAttr({}));
    op->setAttr("self_up_overlap_op", builder.getI64ArrayAttr({}));
    op->setAttr("self_down_overlap_op", builder.getI64ArrayAttr({}));
  });

  // 2.2. update subfunc id & next_index.
  std::vector<func::FuncOp> func_ops;
  submodule.walk([&](func::FuncOp func) {
    if (func.getName() != "main") {
      func_ops.push_back(func);
    }
  });
  for (int id = 0; id < func_ops.size(); ++id) {
    int next_id = id == func_ops.size() - 1 ? -1 : id + 1;
    func_ops[id]->setAttr("id", builder.getI64IntegerAttr(id));
    func_ops[id]->setAttr("next_index",
                          builder.getDenseI32ArrayAttr({next_id}));
  }

  // 2.3 update module info.
  llvm::SmallVector<llvm::StringRef> module_inputs, module_outputs;
  std::vector<Value> inputs, outputs;
  module::getInputsOutputs(submodule, inputs, outputs);
  for (auto &input : inputs) {
    module_inputs.push_back(module::getName(input));
  }
  for (auto &output : outputs) {
    module_outputs.push_back(module::getName(output));
  }
  module::setInputs(module_inputs);
  module::setOutputs(module_outputs);
}
} // namespace funcTool

using namespace moduleTool;
using namespace funcTool;
using namespace groupTool;

class CutFinalMlirPass : public CutFinalMlirBase<CutFinalMlirPass> {
public:
  CutFinalMlirPass() = default;
  void runOnOperation() override {
    assert(module::isBM1684XFamily());
    parse_config_file();
    record_weight_addrs(module::getAllModules()->at(0));
    run(module::getAllModules()->at(0));
  }

private:
  std::vector<llvm::StringRef> original_io_names;
  /// configs
  std::vector<std::string> expected_input_names;
  std::vector<std::string> expected_output_names;
  bool put_storeOp_near_producer_ = true;
  bool remove_unused_local_ops_ = true;
  bool assign_new_io_addrs_ = true;

  /// record weight addrs to minimize the size of bmodel.
  std::set<int64_t> weight_addrs;

private:
  void parse_config_file() {
    auto parseStringArrayAttr = [](const json::Object &obj,
                                   const std::string &key,
                                   std::vector<std::string> &target) {
      if (auto array = obj.getArray(key)) {
        for (const auto &item : *array) {
          if (auto str = item.getAsString()) {
            target.emplace_back(str->str());
          }
        }
      }
    };
    auto parseBoolAttr = [](const json::Object &obj, const std::string &key,
                            bool &target) {
      if (auto val = obj.getBoolean(key))
        target = *val;
    };
    auto bufferOrErr = llvm::MemoryBuffer::getFile(config_file);
    assert(bufferOrErr && "Failed to open file");
    auto jsonOrErr = json::parse((*bufferOrErr)->getBuffer());
    assert(jsonOrErr && "Failed to parse JSON");
    auto &root = *jsonOrErr;
    if (auto *rootObj = root.getAsObject()) {
      parseStringArrayAttr(*rootObj, "new_input_names", expected_input_names);
      parseStringArrayAttr(*rootObj, "new_output_names", expected_output_names);
      parseBoolAttr(*rootObj, "assign_new_io_addrs", assign_new_io_addrs_);
      parseBoolAttr(*rootObj, "remove_unused_local_ops",
                    remove_unused_local_ops_);
      parseBoolAttr(*rootObj, "put_storeop_near_producer",
                    put_storeOp_near_producer_);
    }
  }

  void record_weight_addrs(ModuleOp submodule) {
    submodule.walk(
        [&](top::WeightOp op) { weight_addrs.insert(module::getAddress(op)); });
  }

  void assign_new_addrs_for_new_ios(ModuleOp submodule, bool do_assign) {
    // 1.1 update neuron addr for new ios.
    int64_t addr_limit =
        module::getNeuronAddr(submodule) + module::getNeuronSize(submodule);
    std::vector<Value> ios;
    module::getInputsOutputs(submodule, ios, ios);
    for (auto io_value : ios) {
      if (std::find(original_io_names.begin(), original_io_names.end(),
                    module::getName(io_value)) != original_io_names.end()) {
        continue;
      }
      auto values =
          collectValuesByName(submodule, module::getName(io_value).str(), true);
      if (!do_assign && module::getAddress(values[0]) != 0) {
        continue; // use old address.
      }
      addr_limit = assign_new_addrs(values, addr_limit);
    }
    // update neuron size
    module::setNeuronSize(submodule,
                          addr_limit - module::getNeuronAddr(submodule));
    module::updateModuleTypes();
    /// 1.2 fix addrs for special ops.
    // 1.2.1 ReshapeOps.
    submodule.walk([&](tpu::ReshapeOp op) {
      if (module::getAddress(op.getInput()) ==
          module::getAddress(op.getOutput()))
        return WalkResult::skip();
      int64_t new_addr = std::max(module::getAddress(op.getOutput()),
                                  module::getAddress(op.getInput()));
      auto vals = collectValuesByName(
          submodule, module::getName(op.getInput()).str(), true);
      for (auto val : vals) {
        module::setAddress(val, new_addr);
      }
      vals = collectValuesByName(submodule,
                                 module::getName(op.getOutput()).str(), true);
      for (auto val : vals) {
        module::setAddress(val, new_addr);
      }
      llvm::outs() << "assign new addr for ReshapeOp: "
                   << "\n";
      op.dump();
      return WalkResult::advance();
    });
    // 1.2.2 splitop and joinop
    submodule.walk([&](tpu::CoreParallelOp op) {
      op.walk([&](tpu::CoreSplitOp splitOp) {
        int64_t base_addr = module::getAddress(splitOp->getResult(0));
        int64_t new_base_addr = module::getAddress(op->getOperand(0));
        int64_t offset = new_base_addr - base_addr;
        if (offset != 0) {
          for (auto val : splitOp->getResults()) {
            module::setAddress(val, module::getAddress(val) + offset);
          }
          llvm::outs() << "assign new addr for CoreParallelOp: "
                       << "\n";
          op.dump();
        }
      });
      op.walk([&](tpu::CoreJoinOp joinOp) {
        int64_t base_addr = module::getAddress(joinOp->getOperand(0));
        int64_t new_base_addr = module::getAddress(op->getResult(0));
        int64_t offset = new_base_addr - base_addr;
        if (offset != 0) {
          for (auto val : joinOp->getOperands()) {
            module::setAddress(val, module::getAddress(val) + offset);
          }
          llvm::outs() << "assign new addr for CoreParallelOp: "
                       << "\n";
          op.dump();
        }
      });
    });
    /**
     * Following code only works well for BM1688 and newer chips.
     * TODO: support coeff size minize for BM1684X chip.
     */
    // 2. update coeff size to minimize the size of bmodel.
    // int64_t max_addr = 0;
    // submodule.walk([&](top::WeightOp op) {
    //   max_addr = std::max(max_addr, module::getAddress(op));
    // });
    // auto it = weight_addrs.upper_bound(max_addr);
    // if (it != weight_addrs.end()) {
    //   module::setCoeffSize(submodule, *it - module::getCoeffAddr(submodule));
    // }
    // // update coeff start addr.
    // int64_t weight_start_addr = module::getCoeffAddr(submodule);
    // int64_t new_weight_start_addr =
    //     weight_start_addr + module::getCoeffSize(submodule);
    // submodule.walk([&](top::WeightOp op) {
    //   new_weight_start_addr =
    //       std::min(new_weight_start_addr, module::getAddress(op));
    // });
    // module::setCoeffAddr(submodule, new_weight_start_addr);
    // module::setCoeffSize(submodule, module::getCoeffSize(submodule) +
    //                       new_weight_start_addr - weight_start_addr);

    // 3. update module types again.
    module::updateModuleTypes();
  }

  void run(ModuleOp submodule) {
    original_io_names = get_io_names(submodule); // record.
    // 1. add new inputs as new func-args.
    for (const auto &name : expected_input_names) {
      auto ori_values = collectValuesByName(submodule, name);
      if (!isa<top::InputOp>(ori_values[0].getDefiningOp())) {
        std::sort(ori_values.begin(), ori_values.end(), [&](Value a, Value b) {
          return !isa<tpu::GroupOp>(a.getDefiningOp());
        });
        create_input_op(ori_values[0], submodule);
        augment_sub_funcs_with_new_args(ori_values, submodule);
      }
      if (module::isOpInGroup(ori_values[0].getDefiningOp())) {
        create_load_op(ori_values[0]);
      }
    }
    // 2. add new outputs as new func-rets.
    for (const auto &name : expected_output_names) {
      auto ori_values = collectValuesByName(submodule, name);
      if (module::isOpInGroup(ori_values[0].getDefiningOp())) {
        ori_values.push_back(
            create_store_op(ori_values[0], put_storeOp_near_producer_));
      }
      augment_sub_funcs_with_new_rets(ori_values);
    }
    // 3. update main func with new ios.
    augment_main_func_with_new_ios(submodule, expected_output_names);
    // 4. prune unused vars (local and global).
    prune_unused_vars(submodule, remove_unused_local_ops_);
    // 5. assign new addrs for new ios
    assign_new_addrs_for_new_ios(submodule, assign_new_io_addrs_);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCutFinalMlirPass() {
  return std::make_unique<CutFinalMlirPass>();
}
} // namespace tpu
} // namespace tpu_mlir
