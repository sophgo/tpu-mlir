//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/TruncOp/TruncOp.h"
#include "tpu_mlir/Support/CustomLayer.h"
#include "tpu_mlir/Support/Patterns.h"
using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

class TruncMultiFunc {};

class TruncSingleFunc {};

class TruncLayerPass : public TruncLayerBase<TruncLayerPass> {
public:
  TruncLayerPass() {}

  void _update_ops_io() {
    this->op_ins.clear();
    this->op_outs.clear();

    auto update_io = [&](Operation *op) {
      for (auto in : op->getOperands()) {
        op->dump();
        if (!isa<BlockArgument>(in) && isa<top::NoneOp>(in.getDefiningOp())) {
          continue;
        }
        if (!keep_ops.count(module::getName(in).str())) {
          op_ins[module::getName(in).str()] = in;
          llvm::dbgs() << "op_ins: " << module::getName(in).str()
                       << "; for: " << module::getName(op).str() << "\n";
        }
      }
      if (!module::isOpInBlock(op)) {
        for (auto out : op->getResults()) {
          for (auto dst_op : out.getUsers()) {
            if (!keep_ops.count(module::getName(dst_op).str())) {
              op_outs[module::getName(out).str()] = out;
              llvm::dbgs() << "op_outs: " << module::getName(out).str()
                           << " ; for: " << module::getName(dst_op).str()
                           << "\n";
            }
          }
        }
      }
    };

    for (auto op_pair : keep_ops) {
      auto op = op_pair.second;
      update_io(op);
      if (auto groupOp = dyn_cast<tpu::GroupOp>(op)) {
        groupOp.getBody().walk([&](Operation *op) {
          if (isa<tpu::LoadOp>(op)) {
            update_io(op);
          }
        });
      }
    }
  }

  void _keep_operands(Operation *op) {
    if (module::isOpInGroup(op)) {
      _keep_operands(op->getParentOp());
      return;
    }

    keep_ops[module::getName(op).str()] = op;
    llvm::dbgs() << "keep op: " << module::getName(op).str();
    for (auto opd : op->getOperands()) {
      if (isa<BlockArgument>(opd)) {
        continue;
      }
      if (isa<tpu::ReshapeOp>(opd.getDefiningOp())) {
        _keep_operands(opd.getDefiningOp());
      }
    }
  }

  bool filter_func(FuncOp &func) {
    keep_ops.clear();
    op_ins.clear();
    op_outs.clear();
    auto builder = OpBuilder(func.getContext());

    // step0. collect ops to keep
    func.walk([&](Operation *op) {
      if (cut_locs.count(module::getName(op).str())) {
        _keep_operands(op);
      }
    });

    if (keep_ops.size() == 0) {
      return false;
    }
    _update_ops_io();

    llvm::dbgs() << "============op_ins=============\n";

    // step1. collect erase ops in order
    Operation &lastOp = func.getBody().front().back();
    auto returnOp = dyn_cast<ReturnOp>(lastOp);
    ASSERT_OP(returnOp, &lastOp);

    std::vector<Operation *> removed_ops;
    std::vector<Operation *> removed_ops2;
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa<ReturnOp, FuncOp, tpu::YieldOp, top::NoneOp, top::YieldOp>(op)) {
        auto name = module::getName(op).str();
        if (op_outs.count(name) || keep_ops.count(name)) {
          return;
        }
        if (op_ins.count(name) && isa<top::WeightOp>(op)) {
          return;
        }

        if (module::isOpInBlock(op)) {
          return;
        }
        removed_ops.push_back(op);
      }
    });

    // step2. reset return operands
    std::vector<Value> new_operands;
    for (auto op : op_outs) {
      new_operands.push_back(op.second);
    }
    returnOp->setOperands(new_operands);

    // step1. set op_ins as block arguments
    auto &entryBlock = func.front();
    int oldArgNum = entryBlock.getNumArguments();
    for (auto op : op_ins) {
      if (isa<BlockArgument>(op.second)) {
        llvm::dbgs() << "block arg: "
                     << "; " << module::getName(op.second) << "\n";
      } else {
        llvm::dbgs() << op.second.getDefiningOp()->getName() << "; "
                     << module::getName(op.second) << "\n";
      }
      if (!isa<BlockArgument>(op.second) &&
          isa<top::WeightOp>(op.second.getDefiningOp())) {
        continue;
      }
      auto arg = entryBlock.addArgument(
          op.second.getType(),
          NameLoc::get(builder.getStringAttr(module::getName(op.second))));

      op.second.replaceAllUsesWith(arg);
      llvm::dbgs() << "add arg: " << module::getName(op.second) << "\n";
      op.second.getType().dump();
      op.second.getLoc().dump();
      op.second.dump();
    }
    entryBlock.dump();
    // step3. remove unused ops

    bool erase_some = false;
    while (!removed_ops.empty()) {
      erase_some = false;
      for (auto op : llvm::reverse(removed_ops)) {
        auto name = module::getName(op).str();
        if (op->use_empty()) {
          llvm::dbgs() << "erase op: " << name << ";" << op->getName() << "\n";
          op->erase();
          erase_some = true;
        } else {
          llvm::dbgs() << "erase failed: " << name << ";" << op->getName()
                       << "\n";
          removed_ops2.push_back(op);
        }
      }
      removed_ops = removed_ops2;
      removed_ops2.clear();
      if (removed_ops.size() > 0 && !erase_some) {
        llvm_unreachable("too many ops");
      }
    }
    func.dump();

    for (int i = oldArgNum - 1; i >= 0; --i) {
      entryBlock.eraseArgument(i);
    }

    return true;
  }

  void runOnOperation() override {
    auto mOp = getOperation();
    module::init(mOp);
    std::string input_names = "input_0";
    std::vector<std::string> input_name_list;

    splitString(this->cutLocs, ',', input_name_list);

    for (auto &name : input_name_list) {
      cut_locs.insert(name);
    }

    std::unordered_map<std::string, FuncOp> funcs;
    auto modules = module::getAllModules();
    auto builder = OpBuilder(mOp.getContext());

    auto mainFunc = module::getMainFuncOp((*modules)[0]);
    auto &mainFuncBlock = mainFunc.getBody();
    for (auto s : *modules) {
      for (auto f : s.getOps<FuncOp>()) {
        if (f.getName() == "main") {
          continue;
        }
        // f.dump();
        // TODO, need to support when call results is used by other subfunc
        // in multi subfunc case, function that not hit should be removed
        // directly.
        filter_func(f);
        auto callee = module::getCallOp(f);
        builder.setInsertionPoint(callee);

        std::vector<Value> new_call_ins;
        auto &entryBlock = f.getBody();
        Operation &lastOp = f.getBody().front().back();
        auto returnOp = dyn_cast<ReturnOp>(lastOp);

        auto oldArgNum = mainFuncBlock.getNumArguments();

        for (auto arg : entryBlock.getArguments()) {
          auto blockArg =
              mainFuncBlock.addArgument(arg.getType(), arg.getLoc());
          auto new_arg = builder.create<top::InputOp>(
              arg.getLoc(), arg.getType(), ValueRange{blockArg});
          new_call_ins.push_back(new_arg.getOutput());
        }
        std::vector<Type> new_call_outs;
        for (auto out : returnOp.getOperands()) {
          new_call_outs.push_back(out.getType());
        }

        std::vector<Operation *> old_callee_ipts;
        for (auto op : callee.getOperands()) {
          old_callee_ipts.push_back(op.getDefiningOp());
        }

        // TODO, need to support when call results is used by other subfunc
        auto new_callOp = builder.create<func::CallOp>(
            builder.getUnknownLoc(), f.getName(), new_call_outs, new_call_ins);
        {
          Operation &lastOp = mainFunc.getBody().front().back();
          auto mainReturnOp = dyn_cast<ReturnOp>(lastOp);
          mainReturnOp->setOperands(new_callOp.getResults());
        }

        // callee.replaceAllUsesWith(new_callOp.getResults());
        callee.erase();
        for (auto op : old_callee_ipts) {
          if (op->use_empty()) {
            op->erase();
          }
        }
        for (int i = oldArgNum - 1; i >= 0; --i) {
          mainFuncBlock.eraseArgument(i);
        }
      }
      module::updateModuleTypes();
      // module::getModuleOp().dump();
    }
  }

private:
  // in tensors
  std::unordered_map<std::string, Value> op_ins;
  // out tensors
  std::unordered_map<std::string, Value> op_outs;
  // all op out tensors
  std::unordered_map<std::string, Operation *> keep_ops;
  std::set<std::string> cut_locs;
};

std::unique_ptr<OperationPass<ModuleOp>> createTruncLayerPass() {
  return std::make_unique<TruncLayerPass>();
}
} // namespace tpu
} // namespace tpu_mlir
