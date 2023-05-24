//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

class SubFunction {
public:
  SubFunction(RunMode mode) : mode(mode) {
    count++;
    have_none = false;
  }
  RunMode mode; // tpu/cpu/control
  std::vector<Operation *> ops;
  bool have_none;
  static int count;
};
int SubFunction::count = 0;

void getInputsOutputs(std::vector<Operation *> &ops, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  std::vector<Value> allValues;
  for (auto op : ops) {
    for (auto v : op->getResults()) {
      allValues.push_back(v);
    }
  }
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (v.isa<BlockArgument>()) {
        inputs.push_back(v);
        continue;
      }
      if (find(inputs.begin(), inputs.end(), v) != inputs.end()) {
        continue;
      }
      auto inOp = v.getDefiningOp();
      if (isa<top::NoneOp>(inOp)) {
        continue;
      }
      if (find(allValues.begin(), allValues.end(), v) == allValues.end()) {
        inputs.push_back(v);
      }
    }
    for (auto v : op->getResults()) {
      if (find(outputs.begin(), outputs.end(), v) != outputs.end()) {
        continue;
      }
      for (auto use : v.getUsers()) {
        if (find(ops.begin(), ops.end(), use) == ops.end()) {
          outputs.push_back(v);
          break;
        }
      }
    }
  }

  for (auto &&op : ops) {
    if (isa<tpu::IfOp>(op)) {
      // get the nested's op from above
      for (int i = 0; i < 2; i++) {
        Region &region = op->getRegion(i);
        region.walk([&](Operation *inner_op) {
          for (int k = 0; k < inner_op->getNumOperands(); k++) {
            auto from_op = inner_op->getOperand(k).getDefiningOp();
            if (from_op->getParentOp() != inner_op->getParentOp())
              inputs.emplace_back(inner_op->getOperand(k));
          }
        });
      }
    }
  }
}

void buildSubFunction(std::shared_ptr<SubFunction> sf) {
  // std::vector<Operation *> fnOps;
  std::vector<Value> fnInputs;
  std::vector<Value> fnOutputs;
  getInputsOutputs(sf->ops, fnInputs, fnOutputs);
  std::vector<Type> argType;
  std::vector<Type> resType;
  std::vector<Location> argLoc;
  for (auto input : fnInputs) {
    argType.push_back(input.getType());
    auto ori_input = module::getOriValue(input);
    if (auto op = ori_input.getDefiningOp()) {
      argLoc.push_back(op->getLoc());
    } else {
      argLoc.push_back(module::getLoc());
    }
  }
  for (auto output : fnOutputs) {
    resType.push_back(output.getType());
  }
  int64_t id = SubFunction::count - 1;
  std::string func_name = "subfunc_" + std::to_string(id);
  OpBuilder builder(module::getCtx());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("id", builder.getI64IntegerAttr(id)));
  attrs.push_back(builder.getNamedAttr(
      "mode", RunModeAttr::get(module::getCtx(), sf->mode)));
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{argType},
                                        llvm::ArrayRef<Type>{resType});
  auto fnOp = FuncOp::create(module::getLoc(), func_name, fnType,
                             ArrayRef<NamedAttribute>(attrs));
  auto block = fnOp.addEntryBlock();
  builder.setInsertionPointAfterValue(fnOutputs.back());
  func::CallOp callOp = builder.create<func::CallOp>(
      module::getLoc(), func_name, resType, fnInputs);
  for (auto it : llvm::enumerate(callOp.getResults())) {
    fnOutputs[it.index()].replaceUsesWithIf(
        it.value(), [&](OpOperand &operand) {
          Operation *user = operand.getOwner();
          return find(sf->ops.begin(), sf->ops.end(), user) == sf->ops.end();
        });
  }
  builder.setInsertionPointToStart(block);
  top::NoneOp noneOp;
  if (sf->have_none) {
    noneOp =
        builder.create<top::NoneOp>(module::getLoc(), builder.getNoneType());
  }
  auto retOp = builder.create<ReturnOp>(module::getLoc(), fnOutputs);
  for (auto op : sf->ops) {
    if (isa<top::NoneOp>(op)) {
      continue;
    }
    for (auto it : llvm::enumerate(op->getOperands())) {
      if (!it.value().isa<BlockArgument>() &&
          isa<top::NoneOp>(it.value().getDefiningOp())) {
        op->setOperand(it.index(), noneOp);
      }
    }
    op->moveBefore(retOp);
  }
  module::push_back(fnOp);
  for (auto it : llvm::enumerate(fnInputs)) {
    auto arg = block->getArgument(it.index());
    arg.setLoc(argLoc[it.index()]);
    it.value().replaceUsesWithIf(arg, [&](OpOperand &operand) {
      /* bugfix:according to the value's def-use,
         check the proper ancestor's operand's owner */
      return fnOp->isProperAncestor(operand.getOwner());
    });
  }
}

static void insert_subop(std::shared_ptr<SubFunction> &subf, Operation *op) {
  for (auto opd : op->getOperands()) {
    if (!opd.isa<BlockArgument>()) {
      auto op_ = opd.getDefiningOp();
      if (isa<top::WeightOp>(op_)) {
        if (std::find(subf->ops.begin(), subf->ops.end(), op_) ==
            subf->ops.end()) {
          subf->ops.push_back(op_);
        }
      } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
        subf->have_none = true;
      }
    }
  }
  subf->ops.push_back(op);
}

class SubnetDividePass : public SubnetDivideBase<SubnetDividePass> {
public:
  SubnetDividePass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_REORDERED)) {
      llvm_unreachable("module should be reordered");
    }
    divide_func();
    module::removeUnusedOp();
    module::setState(module::State::TPU_DIVIDED);
  }

  static bool force_dynamic_run(Operation *op) {
    if (isa<TopKOp, YoloDetectionOp, DetectionOutputOp, RoiAlignOp, NonZeroOp>(
            op)) {
      return true;
    } else if (op->hasTrait<trait::ShapeProducer>()) {
      return true;
    } else if (op->hasTrait<trait::ShapeConsumer>()) {
      return true;
    } else if (isa<SliceOp>(op)) {
      return !module::isNone(dyn_cast<SliceOp>(op).getOffsetT());
    } else if (module::isBM1684Family()) {
      if (auto gather_op = dyn_cast<tpu::GatherOp>(op)) {
        if (!module::isWeight(gather_op.getIndices()))
          return true;
      }
    }
    return false;
  }

  // seperate: whether seperate with other op
  RunMode getOpMode(Operation *op, bool &seperate) {
    seperate = false;
    if (isa<GenericCpuOp>(op)) {
      seperate = true;
      return RunMode::CPU;
    } else if (isa<tpu::IfOp>(op)) {
      seperate = true;
      return RunMode::SWITCH;
    } else if (isa<tpu::YieldOp>(op)) {
      seperate = true;
      return RunMode::UNKNOW;
    } else if (dynamic || force_dynamic_run(op)) {
      return RunMode::TPU_DYNAMIC;
    }
    return RunMode::TPU_STATIC;
  }

  bool toposortAction(Block *block, llvm::iterator_range<Block::iterator> ops) {
    auto isOpReady = [&](Operation *op, llvm::DenseSet<Operation *> &unscheduledOps) -> bool {
        // An operation is ready to be scheduled if all its operands are ready. An
        const auto isReady = [&](Value value) {
          Operation *parent = value.getDefiningOp();
          if (!parent)
            return true;
          do {
            if (parent == op)
              return true;
            if (unscheduledOps.contains(parent))
              return false;
          } while ((parent = parent->getParentOp()));
          return true;
        };

        WalkResult readyToSchedule = op->walk([&](Operation *nestedOp) {
          return llvm::all_of(nestedOp->getOperands(),
                            [&](Value operand) { return isReady(operand); })
                  ? WalkResult::advance()
                  : WalkResult::interrupt();
      });
      return !readyToSchedule.wasInterrupted();
    };

    llvm::DenseSet<Operation *> unscheduledOps;
    for (Operation &op : ops)
      unscheduledOps.insert(&op);

    Block::iterator nextScheduledOp = ops.begin();
    Block::iterator end = ops.end();

    bool allOpsScheduled = true;
    while (!unscheduledOps.empty()) {
      bool scheduledAtLeastOnce = false;

      for (Operation &op :
          llvm::make_early_inc_range(llvm::make_range(nextScheduledOp, end))) {
        if (!isOpReady(&op, unscheduledOps))
          continue;

        unscheduledOps.erase(&op);
        op.moveBefore(block, nextScheduledOp);
        scheduledAtLeastOnce = true;
        if (&op == &*nextScheduledOp)
          ++nextScheduledOp;
      }

      if (!scheduledAtLeastOnce) {
        allOpsScheduled = false;
        unscheduledOps.erase(&*nextScheduledOp);
        ++nextScheduledOp;
      }
    }

    return allOpsScheduled;
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

  void divide_func() {
    auto mainFunc = module::getMainFuncOp();
    std::shared_ptr<SubFunction> subf = nullptr;
    bool seperate;
    // for to traverse the nested regions, walk by preorder preferred.
    mainFunc.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<top::InputOp, top::WeightOp, FuncOp, top::NoneOp, ReturnOp,
              func::CallOp>(op)) {
        // do nothing
      } else {
        auto mode = getOpMode(op, seperate);
        if (seperate) {
          if (subf != nullptr) {
            buildSubFunction(subf);
          }

          if (mode != RunMode::UNKNOW) {
            subf = std::make_shared<SubFunction>(mode);
            insert_subop(subf, op);
            buildSubFunction(subf);
          }
          subf = nullptr;
        } else if (subf == nullptr) {
          subf = std::make_shared<SubFunction>(mode);
          insert_subop(subf, op);
        } else if (subf->mode == mode) {
          insert_subop(subf, op);
        } else {
          buildSubFunction(subf);
          subf = std::make_shared<SubFunction>(mode);
          insert_subop(subf, op);
        }
      }
    });
    if (subf != nullptr) {
      buildSubFunction(subf);
      subf = nullptr;
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass() {
  return std::make_unique<SubnetDividePass>();
}
} // namespace tpu
} // namespace tpu_mlir
