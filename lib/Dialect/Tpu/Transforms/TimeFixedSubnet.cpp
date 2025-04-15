//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace tpu_mlir {
namespace tpu {

struct timefixed_subnet_info;
using InfoVec = llvm::SmallVector<timefixed_subnet_info *>;
struct timefixed_subnet_info {
  timefixed_subnet_info() : id(__next_id) {
    index = -1;
    __next_id++;
  }
  static void reset_id() { __next_id = 0; }
  std::vector<Operation *> ops;
  std::vector<Value> ins;
  std::vector<Value> outs;
  const int id;
  int index;
  static int __next_id;
  InfoVec next_subnets;
  InfoVec prev_subnets;
  std::vector<int> next_index;
};
int timefixed_subnet_info::__next_id = 0;

class TimeFixedSubnetPass : public TimeFixedSubnetBase<TimeFixedSubnetPass> {
public:
  TimeFixedSubnetPass() = default;
  void runOnOperation() override {
    std::string jsonfile = this->json_file;
    auto &ctx = getContext();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      // only support one subfunc, and the subfunc_0 must a static subfunc
      if (!valid_module(s))
        llvm_unreachable("Unsupported module!");
      InfoVec subnet_infos = base_subnet_split(s, jsonfile);
      auto sorted_subnet_infos = sort_subnets(subnet_infos);
      sorted_subnet_infos.back()->next_index.assign({-1});
      reconstruct_ir(sorted_subnet_infos, s);
      merge_block(s);
      for (auto info : sorted_subnet_infos)
        delete info;
      toposort();
      for (auto func : s.getOps<FuncOp>()) {
        RewritePatternSet patterns(&ctx);
        applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
    }
    module::removeUnusedOp();
    module::updateModuleTypes();
  }

  bool valid_module(ModuleOp module) {
    FuncOp subfunc_0 = nullptr;
    int subfunc_num = 0;
    for (auto func : module.getOps<FuncOp>()) {
      if (func.getName().startswith("subfunc_")) {
        subfunc_num++;
        if (func.getName() == "subfunc_0")
          subfunc_0 = func;
      }
    }
    if (subfunc_num == 1) {
      auto mode_attr = subfunc_0->getAttrOfType<RunModeAttr>("mode");
      return mode_attr.getValue() == RunMode::TPU_STATIC;
    }
    return false;
  }

  InfoVec base_subnet_split(ModuleOp sub, std::string json) {
    timefixed_subnet_info::reset_id();
    InfoVec subnet_infos;
    std::vector<Operation *> all_ops;
    llvm::DenseSet<Value> valid_values;
    auto mainfunc = module::getFuncOp(sub, "subfunc_0");

    mainfunc.walk<WalkOrder::PreOrder, ForwardDominanceIterator<true>>(
        [&](Operation *op) {
          if (isa<func::FuncOp, ReturnOp, top::NoneOp>(op))
            return WalkResult::advance();
          else if (isa<top::WeightOp>(op)) {
            valid_values.insert(op->getResult(0));
          } else if (isa<tpu::GroupOp>(op)) {
            llvm::DenseSet<Value> insertedOperands(op->getOperands().begin(),
                                                   op->getOperands().end());
            llvm::SmallVector<Value> new_operands;
            llvm::DenseMap<Value, Value> operand_mapping;
            op->walk([&](Operation *innerOp) {
              if (isa<tpu::LoadOp>(innerOp)) {
                for (Value operand : innerOp->getOperands()) {
                  if ((operand.getDefiningOp() &&
                       isa<top::WeightOp>(operand.getDefiningOp())) ||
                      isa<BlockArgument>(operand)) {
                    if (insertedOperands.insert(operand).second) {
                      new_operands.emplace_back(operand);
                      operand_mapping[operand] = operand;
                    }
                  }
                }
              }
            });
            if (!new_operands.empty()) {
              OpBuilder builder(op);
              op->insertOperands(op->getNumOperands(), new_operands);
              op->walk([&](Operation *innerOp) {
                if (isa<tpu::LoadOp>(innerOp)) {
                  for (unsigned i = 0; i < innerOp->getNumOperands(); ++i) {
                    Value operand = innerOp->getOperand(i);
                    if (operand_mapping.count(operand))
                      innerOp->setOperand(i, operand_mapping[operand]);
                  }
                }
              });
            }
            all_ops.emplace_back(op);
            return WalkResult::skip();
          }
          all_ops.emplace_back(op);
          return WalkResult::advance();
        });
    std::vector<std::string> loc_info = get_json_info(json);
    auto dfs = [&]() noexcept -> void {
      while (!all_ops.empty()) {
        subnet_infos.emplace_back(new timefixed_subnet_info);
        auto &info = subnet_infos.back();
        bool subnet_ended = false;
        while (!subnet_ended && !all_ops.empty()) {
          bool updated = false;
          auto it = all_ops.begin();
          while (it != all_ops.end()) {
            auto op = *it;
            if (op_can_run(op, valid_values)) {
              info->ops.emplace_back(op);
              valid_values.insert(op->result_begin(), op->result_end());
              it = all_ops.erase(it);
              updated = true;
              if (should_end_subnet(op, loc_info)) {
                subnet_ended = true;
                break;
              }
            } else {
              ++it;
            }
          }
          if (!updated)
            break;
        }
      }
    };
    dfs();
    return subnet_infos;
  }

  std::vector<std::string> get_json_info(const std::string &json_file) {
    std::vector<std::string> result;
    auto fileOrError = MemoryBuffer::getFile(json_file);
    if (!fileOrError) {
      llvm_unreachable("Error opening JSON file!");
    }
    auto jsonVal = json::parse((*fileOrError)->getBuffer());
    if (!jsonVal)
      return result;
    const auto subfuncs = jsonVal->getAsObject()->getArray("subfuncs");
    if (!subfuncs)
      return result;
    for (const auto &subfunc : *subfuncs) {
      if (const auto obj = subfunc.getAsObject()) {
        if (auto locVal = obj->get("last_loc")) {
          if (auto locStr = locVal->getAsString()) {
            result.emplace_back(locStr->str());
            continue;
          }
        }
      }
    }
    return result;
  }

  bool op_can_run(Operation *op, const llvm::DenseSet<Value> &valid_values) {
    for (auto operand : op->getOperands()) {
      Operation *parent = operand.getDefiningOp();
      if (parent && !isa<top::WeightOp, top::NoneOp>(parent) &&
          !valid_values.count(operand)) {
        return false;
      }
    }
    return true;
  }

  bool should_end_subnet(Operation *op, std::vector<std::string> &loc_info) {
    assert(!loc_info.empty() && "Empty location info!");
    if (loc_info.size() == 1 && loc_info.front() == "func.return")
      return false;
    if (hasNameLoc(op->getLoc(), loc_info.front())) {
      loc_info.erase(loc_info.begin());
      return true;
    }
    return false;
  }

  static bool hasNameLoc(Location loc, StringRef target) {
    if (isa<UnknownLoc>(loc))
      return false;
    if (auto nameLoc = loc.dyn_cast<NameLoc>())
      return nameLoc.getName() == target;
    if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      for (Location sub : fusedLoc.getLocations())
        if (hasNameLoc(sub, target))
          return true;
    }
    return false;
  }

  InfoVec sort_subnets(InfoVec &subnet_infos) {
    gen_subnet_data_depence(subnet_infos);
    InfoVec sorted_subnets;
    for (auto &subnet : subnet_infos) {
      if (subnet_can_run(subnet, sorted_subnets)) {
        subnet->index = sorted_subnets.size();
        subnet->next_index.emplace_back(subnet->index + 1);
        sorted_subnets.emplace_back(subnet);
      } else {
        llvm_unreachable("Subnet dependency not met");
      }
    }
    sorted_subnets.back()->next_index.clear();
    return sorted_subnets;
  }

  void gen_subnet_data_depence(InfoVec &subnet_infos) {
    llvm::DenseMap<Value, timefixed_subnet_info *> value_from_subnets;
    for (const auto &info : subnet_infos) {
      info->prev_subnets.clear();
      info->next_subnets.clear();
      for (auto &&out : info->outs) {
        value_from_subnets[out] = info;
      }
    }

    for (const auto &info : subnet_infos) {
      for (auto &&in : info->ins) {
        auto from_subnet = value_from_subnets[in];
        if (from_subnet) {
          from_subnet->next_subnets.emplace_back(info);
          info->prev_subnets.emplace_back(from_subnet);
        }
      }
    }
    return;
  }

  bool subnet_can_run(timefixed_subnet_info *subnet, const InfoVec &run_group) {
    for (auto &in : subnet->prev_subnets) {
      if (std::find(run_group.begin(), run_group.end(), in) == run_group.end())
        return false;
    }
    return true;
  }

  void reconstruct_ir(InfoVec &subnets, ModuleOp submodule) {
    for (auto &subnet : subnets) {
      auto *origBlock = subnet->ops.front()->getBlock();
      auto insertAfter = std::next(subnet->ops.back()->getIterator());
      std::vector<Value> fnInputs, fnOutputs;
      bool has_NoneOp = false;
      getInputsOutputs(subnet->ops, fnInputs, fnOutputs, has_NoneOp);

      llvm::DenseSet<Value> inputSet(fnInputs.begin(), fnInputs.end());
      fnOutputs.erase(
          std::remove_if(fnOutputs.begin(), fnOutputs.end(),
                         [&](Value v) { return inputSet.contains(v); }),
          fnOutputs.end());
      SmallVector<Type> argTypes, resTypes;
      SmallVector<Location> argLocs;
      OpBuilder builder(module::getCtx());
      OpBuilder::InsertionGuard insertGuard(builder);
      for (auto v : fnInputs) {
        argTypes.emplace_back(v.getType());
        auto ori = module::getOriValue(v);
        argLocs.emplace_back(module::isNone(ori) ? module::getLoc()
                                                 : module::getLoc(ori));
      }
      for (auto v : fnOutputs)
        resTypes.emplace_back(v.getType());

      auto ctx = module::getCtx();
      std::string funcName = "mainfunc_" + std::to_string(subnet->index);
      auto fnType = builder.getFunctionType(argTypes, resTypes);
      auto fnOp = FuncOp::create(
          module::getLoc(), funcName, fnType,
          ArrayRef<NamedAttribute>{
              builder.getNamedAttr("id",
                                   builder.getI64IntegerAttr(subnet->index)),
              builder.getNamedAttr("mode",
                                   RunModeAttr::get(ctx, RunMode::TPU_STATIC)),
              builder.getNamedAttr("next_index", builder.getDenseI32ArrayAttr(
                                                     subnet->next_index))});
      auto *block = fnOp.addEntryBlock();
      llvm::DenseMap<Value, Value> val2Arg;
      for (size_t i = 0; i < fnInputs.size(); ++i) {
        auto arg = block->getArgument(i);
        arg.setLoc(argLocs[i]);
        val2Arg[fnInputs[i]] = arg;
      }

      builder.setInsertionPointToStart(block);
      top::NoneOp noneOp;
      if (has_NoneOp)
        noneOp = builder.create<top::NoneOp>(module::getLoc(),
                                             builder.getNoneType());

      auto patchOperands = [&](Operation *op) {
        for (auto &operand : op->getOpOperands()) {
          auto it = val2Arg.find(operand.get());
          if (it != val2Arg.end()) {
            operand.set(it->second);
          } else if (has_NoneOp && isa_and_nonnull<top::NoneOp>(
                                       operand.get().getDefiningOp())) {
            operand.set(noneOp);
          }
        }
      };

      for (auto *op : subnet->ops) {
        op->moveBefore(block, block->end());
        op->walk([&](Operation *nested) { patchOperands(nested); });
      }

      block->walk([&](tpu::GroupOp g) { isolateGroupOp(g); });
      builder.setInsertionPointToEnd(block);
      builder.create<ReturnOp>(module::getLoc(), fnOutputs);
      submodule.push_back(fnOp);
      builder.setInsertionPoint(origBlock, insertAfter);
      auto callOp = builder.create<func::CallOp>(module::getLoc(), funcName,
                                                 resTypes, fnInputs);

      for (auto v : llvm::enumerate(fnOutputs)) {
        Value oldVal = v.value();
        Value newVal = callOp.getResult(v.index());
        oldVal.replaceUsesWithIf(newVal, [&](OpOperand &use) {
          return !fnOp->isAncestor(use.getOwner());
        });
      }
    }
  }

  void getInputsOutputs(std::vector<Operation *> &ops,
                        std::vector<Value> &inputs, std::vector<Value> &outputs,
                        bool &has_NoneOp) {
    has_NoneOp = false;
    llvm::DenseSet<Value> internalValues;
    llvm::DenseSet<Operation *> internalOps;
    llvm::DenseSet<Value> inputSet;
    llvm::DenseSet<Value> outputSet;
    auto collectInternal = [&](Operation *op) {
      internalOps.insert(op);
      for (auto v : op->getResults()) {
        internalValues.insert(v);
      }
    };
    for (Operation *op : ops) {
      collectInternal(op);
      if (auto groupOp = dyn_cast<tpu::GroupOp>(op)) {
        for (Operation &inner : groupOp.getBody().front()) {
          collectInternal(&inner);
        }
      }
    }
    auto processOperand = [&](Value operand) {
      if (operand.isa<BlockArgument>()) {
        if (inputSet.insert(operand).second) {
          inputs.push_back(operand);
        }
        return;
      }
      if (isa<top::NoneOp>(operand.getDefiningOp())) {
        has_NoneOp = true;
        return;
      }

      if (!internalValues.contains(operand) &&
          inputSet.insert(operand).second) {
        inputs.push_back(operand);
      }
    };

    for (Operation *op : ops) {
      for (auto operand : op->getOperands()) {
        processOperand(operand);
      }
      if (auto groupOp = dyn_cast<tpu::GroupOp>(op)) {
        for (Operation &inner : groupOp.getBody().front()) {
          if (auto load = dyn_cast<tpu::LoadOp>(&inner)) {
            auto src = load.getInput();
            if (!internalValues.contains(src) && inputSet.insert(src).second) {
              inputs.push_back(src);
            }
          }
        }
      }
      for (auto result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!internalOps.contains(user) && outputSet.insert(result).second) {
            outputs.push_back(result);
            break;
          }
        }
      }
    }
    llvm::sort(inputs, [](Value a, Value b) {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    });
  }

  static void isolateGroupOp(tpu::GroupOp groupOp) {
    auto &body = groupOp.getBody().front();
    OpBuilder builder(groupOp);
    llvm::DenseMap<Value, Value> val2Arg;
    unsigned NumArgs = body.getNumArguments();
    for (unsigned i = 0; i < NumArgs; ++i)
      if (!body.getArgument(i).getType().isa<mlir::NoneType>())
        val2Arg.try_emplace(groupOp->getOperand(i), body.getArgument(i));

    SmallVector<Value> newOperands;
    llvm::DenseSet<Value> alreadyAdded;
    auto addOperand = [&](Value v, BlockArgument barg) {
      if (alreadyAdded.insert(v).second)
        newOperands.emplace_back(v);
      val2Arg[v] = barg;
    };
    body.walk([&](Operation *op) {
      for (OpOperand &opd : op->getOpOperands()) {
        auto v = opd.get();
        if (v.getParentBlock() == &body)
          continue;
        if (v.getType().isa<mlir::NoneType>()) {
          auto it = val2Arg.find(v);
          if (it == val2Arg.end()) {
            builder.setInsertionPointToStart(&body);
            auto NoneOp = builder.create<top::NoneOp>(module::getLoc(),
                                                      builder.getNoneType());
            val2Arg[v] = NoneOp.getResult();
          }
          opd.set(val2Arg[v]);
          continue;
        }
        auto it = val2Arg.find(v);
        if (it == val2Arg.end()) {
          auto loc = v.getLoc();
          if (auto opResult = v.dyn_cast<mlir::OpResult>()) {
            unsigned resultIndex = opResult.getResultNumber();
            if (auto fusedLoc = loc.dyn_cast<mlir::FusedLoc>()) {
              auto subLocs = fusedLoc.getLocations();
              loc = subLocs[resultIndex];
            }
          }
          BlockArgument barg = body.addArgument(v.getType(), loc);
          addOperand(v, barg);
        }
        opd.set(val2Arg[v]);
      }
    });
    groupOp->setOperands(newOperands);
  }

  // merge main and subfunc, rename mainfunc to subfunc
  void merge_block(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    SymbolTable symTable(module);
    SmallVector<std::pair<FuncOp, std::string>> rename;
    for (FuncOp f : module.getOps<FuncOp>()) {
      StringRef name = f.getSymName();
      if (name.consume_front("mainfunc_"))
        rename.emplace_back(f, ("subfunc_" + name).str());
    }
    for (auto &[func, newName] : rename) {
      if (failed(SymbolTable::replaceAllSymbolUses(
              func, StringAttr::get(ctx, newName), module)))
        llvm::report_fatal_error("symbol replace failed");
      SymbolTable::setSymbolName(func, newName);
    }
    FuncOp main = module.lookupSymbol<FuncOp>("main");
    if (!main)
      return;
    CallOp targetCall;
    main.walk([&](CallOp c) {
      if (c.getCallee() == "subfunc_0")
        targetCall = c;
    });
    if (!targetCall)
      return;
    FuncOp subfunc0 = module.lookupSymbol<FuncOp>("subfunc_0");
    if (!subfunc0)
      return;
    inlineSingleCall(targetCall, subfunc0);
    if (subfunc0.use_empty())
      subfunc0.erase();
  }

  static void inlineSingleCall(CallOp call, FuncOp callee) {
    OpBuilder b(call);
    IRMapping map;
    for (auto it : llvm::enumerate(call.getOperands()))
      map.map(callee.getArgument(it.index()), it.value());
    SmallVector<Operation *> toClone;
    for (Operation &op : callee.getBody().front())
      if (!isa<func::ReturnOp>(op))
        toClone.push_back(&op);
    for (Operation *op : toClone)
      b.clone(*op, map);
    auto ret = cast<func::ReturnOp>(callee.getBody().front().getTerminator());
    for (auto zip : llvm::zip(call.getResults(), ret.getOperands()))
      std::get<0>(zip).replaceAllUsesWith(map.lookup(std::get<1>(zip)));
    call.erase();
  }

  void toposort() {
    module::getModuleOp().walk([&](Operation *op) {
      for (auto it : llvm::enumerate(op->getRegions())) {
        for (Block &block : it.value()) {
          if (block.empty())
            continue;
          if (block.back().hasTrait<OpTrait::IsTerminator>())
            toposortAction(&block, block.without_terminator());
          else
            toposortAction(&block,
                           llvm::make_range(block.begin(), block.end()));
        }
      }
    });
  }

  bool toposortAction(Block *block, llvm::iterator_range<Block::iterator> ops) {
    if (ops.empty())
      return true;
    auto isOpReady = [](Operation *op,
                        const llvm::DenseSet<Operation *> &unscheduledOps) {
      auto isOperandReady = [&](Value value) {
        for (Operation *parent = value.getDefiningOp(); parent;
             parent = parent->getParentOp()) {
          if (parent == op)
            return true;
          if (unscheduledOps.contains(parent))
            return false;
        }
        return true;
      };
      return op->walk([&](Operation *nestedOp) {
                 return llvm::all_of(nestedOp->getOperands(), isOperandReady)
                            ? WalkResult::advance()
                            : WalkResult::interrupt();
               }).wasInterrupted() == false;
    };

    llvm::DenseSet<Operation *> unscheduledOps;
    for (Operation &op : ops)
      unscheduledOps.insert(&op);
    auto nextScheduledOp = ops.begin();
    bool allOpsScheduled = true;
    while (!unscheduledOps.empty()) {
      bool scheduledAny = false;
      for (Operation &op : llvm::make_early_inc_range(
               llvm::make_range(nextScheduledOp, ops.end()))) {
        if (!isOpReady(&op, unscheduledOps))
          continue;

        unscheduledOps.erase(&op);
        op.moveBefore(block, nextScheduledOp);
        scheduledAny = true;
        if (&op == &*nextScheduledOp)
          ++nextScheduledOp;
      }

      if (!scheduledAny) {
        allOpsScheduled = false;
        unscheduledOps.erase(&*nextScheduledOp);
        ++nextScheduledOp;
      }
    }
    return allOpsScheduled;
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createTimeFixedSubnetPass() {
  return std::make_unique<TimeFixedSubnetPass>();
}

} // namespace tpu
} // namespace tpu_mlir
