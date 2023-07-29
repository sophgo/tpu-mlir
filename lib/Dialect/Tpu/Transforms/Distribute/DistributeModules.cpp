//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Distribute/Distribute.h"

namespace tpu_mlir {
namespace tpu {

class SubFunction {
public:
  SubFunction(int devid) : devid(devid) {
    count++;
    have_none = false;
  }
  int devid;
  std::vector<Operation *> ops;
  bool have_none;
  static int count;
};

int SubFunction::count = 0;

static void getInputsOutputs(std::vector<Operation *> &ops,
                             std::vector<Value> &inputs,
                             std::vector<Value> &outputs) {
  std::vector<Value> allValues;
  for (auto op : ops) {
    for (auto v : op->getResults()) {
      allValues.push_back(v);
    }
  }
  for (auto op : ops) {
    for (auto v : op->getOperands()) {
      if (find(inputs.begin(), inputs.end(), v) != inputs.end()) {
        continue;
      }
      auto inOp = v.getDefiningOp();
      if (inOp == nullptr || isa<top::NoneOp>(inOp)) {
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
}

static void buildSubFunction(std::shared_ptr<SubFunction> sf, ModuleOp module) {
  if (sf == nullptr || sf->ops.empty()) {
    return;
  }
  // std::vector<Operation *> fnOps;
  std::vector<Value> fnInputs;
  std::vector<Value> fnOutputs;
  getInputsOutputs(sf->ops, fnInputs, fnOutputs);
  std::vector<Type> argType;
  std::vector<Type> resType;
  std::vector<Location> argLoc;
  for (auto input : fnInputs) {
    argType.push_back(input.getType());
    if (auto op = input.getDefiningOp()) {
      argLoc.push_back(op->getLoc());
    } else {
      argLoc.push_back(module.getLoc());
    }
  }
  for (auto output : fnOutputs) {
    resType.push_back(output.getType());
  }
  int64_t id = SubFunction::count - 1;
  std::string func_name = "subfunc_" + std::to_string(id);
  OpBuilder builder(module.getContext());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("id", builder.getI64IntegerAttr(id)));
  attrs.push_back(
      builder.getNamedAttr("device_id", builder.getI64IntegerAttr(sf->devid)));
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{argType},
                                        llvm::ArrayRef<Type>{resType});
  auto fnOp = FuncOp::create(module.getLoc(), func_name, fnType,
                             ArrayRef<NamedAttribute>(attrs));
  auto block = fnOp.addEntryBlock();
  builder.setInsertionPointAfterValue(fnOutputs.back());
  func::CallOp callOp = builder.create<func::CallOp>(module.getLoc(), func_name,
                                                     resType, fnInputs);
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
        builder.create<top::NoneOp>(module.getLoc(), builder.getNoneType());
  }
  auto retOp = builder.create<func::ReturnOp>(module.getLoc(), fnOutputs);
  for (auto op : sf->ops) {
    if (isa<top::NoneOp>(op)) {
      continue;
    }
    for (auto it : llvm::enumerate(op->getOperands())) {
      if (isa_and_nonnull<top::NoneOp>(it.value().getDefiningOp())) {
        op->setOperand(it.index(), noneOp);
      }
    }
    op->moveBefore(retOp);
  }
  module.push_back(fnOp);
  for (auto it : llvm::enumerate(fnInputs)) {
    auto arg = block->getArgument(it.index());
    arg.setLoc(argLoc[it.index()]);
    it.value().replaceUsesWithIf(arg, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return find(sf->ops.begin(), sf->ops.end(), user) != sf->ops.end();
    });
  }
}

static void insert_subop(std::shared_ptr<SubFunction> &subf, Operation *op) {
  for (auto opd : op->getOperands()) {
    auto op_ = opd.getDefiningOp();
    if (isa<top::WeightOp>(op_)) {
      subf->ops.push_back(op_);
    } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
      subf->have_none = true;
    }
  }
  subf->ops.push_back(op);
}

// MatMulSliceMerge use backward
static void collect_ops_backward(std::shared_ptr<SubFunction> &subf,
                                 Operation *op) {
  for (int i = 0; i < op->getNumOperands(); i++) {
    auto opd = op->getOperand(i);
    auto op_ = opd.getDefiningOp();
    if (isa<tpu::DistributionBeginOp>(op_)) {
      op->setOperand(i, op_->getOperand(0));
      continue;
    } else if (!isa<top::WeightOp, top::NoneOp>(op_)) {
      collect_ops_backward(subf, op_);
    } else if (isa<top::WeightOp>(op_)) {
      subf->ops.push_back(op_);
    } else if (isa<top::NoneOp>(op_) && subf->have_none == false) {
      subf->have_none = true;
    }
  }
  subf->ops.push_back(op);
}

// MatMulTopK use forward
static void Scollect_ops_forward(std::shared_ptr<SubFunction> &subf,
                                 Operation *op) {
  auto op_ = op->getOperand(0).getDefiningOp();
  if (isa<tpu::DistributionBeginOp>(op_)) {
    op->setOperand(0, op_->getOperand(0));
  }
  insert_subop(subf, op);
  for (auto u : op->getUsers()) {
    if (isa<tpu::DistributionEndOp>(u)) {
      continue;
    }
    Scollect_ops_forward(subf, u);
  }
}

static void buildDistibution(tpu::DistributionBeginOp begin,
                             tpu::DistributionEndOp end, ModuleOp m,
                             int64_t num_devices) {
  std::vector<Operation *> begins(begin->user_begin(), begin->user_end());
  std::vector<Value> ends(end->operand_begin(), end->operand_end());
  for (int i = 0; i < num_devices; i++) {
    auto subf = std::make_shared<SubFunction>(i);
    if (begins.size() == num_devices) {
      Scollect_ops_forward(subf, begins[i]);
    } else if (ends.size() == num_devices) {
      collect_ops_backward(subf, ends[i].getDefiningOp());
    } else {
      llvm_unreachable("Not Implemented");
    }
    buildSubFunction(subf, m);
  }
}

class EndOpTranslate : public OpRewritePattern<tpu::DistributionEndOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::DistributionEndOp op,
                                PatternRewriter &rewriter) const override {
    auto num_devices = module::getDeviceNum();
    rewriter.setInsertionPointAfter(op);
    switch (op.getPattern()) {
    case tpu::DistributionPattern::MatMulSliceMerge: {
      // convert to AddOp
      auto output = op.getOperand(0);
      for (int i = 1; i < num_devices; i++) {
        auto loc = module::getLocLike(op.getOutput(), std::to_string(i));
        auto add = rewriter.create<tpu::AddOp>(
            loc, op.getOutput().getType(),
            mlir::ValueRange{output, op.getOperand(i)});
        output = add.getOutput();
      }
      rewriter.replaceOp(op, {output});
    } break;
    case tpu::DistributionPattern::MatMulTopK:
      auto value = op.getOperand(0);
      auto indice = op.getOperand(1);
      for (int i = 1; i < num_devices; i++) {
        auto value2 = op.getOperand(i * 2);
        auto indice2 = op.getOperand(i * 2 + 1);
        auto loc =
            module::getLocLike(op.getOutput(), "add" + std::to_string(i));
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr(
            "mode", rewriter.getStringAttr("GreaterOrEqual")));
        auto cmp = rewriter.create<tpu::CompareOp>(
            loc, value.getType(), ValueRange{value, value2}, attrs);
        loc = module::getLocLike(op.getOutput(), "value" + std::to_string(i));
        auto value_select = rewriter.create<tpu::WhereOp>(
            loc, value.getType(), ValueRange{cmp.getOutput(), value, value2});
        value = value_select.getOutput();
        loc = module::getLocLike(op.getOutput(), "indice" + std::to_string(i));
        auto indice_select = rewriter.create<tpu::WhereOp>(
            loc, indice.getType(),
            ValueRange{cmp.getOutput(), indice, indice2});
        indice = indice_select.getOutput();
      }
      rewriter.replaceOp(op, {indice});
      break;
    }
    return success();
  }
};

void DistributeModules(ModuleOp m, int64_t num_device) {
  auto main = module::getMainFuncOp();
  std::shared_ptr<SubFunction> subf = nullptr;
  bool in_distribution = false;
  tpu::DistributionBeginOp begin;
  main.walk([&](Operation *op) {
    if (isa<top::InputOp, top::WeightOp, FuncOp, top::NoneOp, func::ReturnOp,
            func::CallOp>(op)) {
      // do nothing
    } else {
      if (isa<tpu::DistributionBeginOp>(op)) {
        // for some patterns maybe do slice here
        buildSubFunction(subf, m);
        in_distribution = true;
        begin = cast<tpu::DistributionBeginOp>(op);
      } else if (isa<tpu::DistributionEndOp>(op)) {
        auto end = cast<tpu::DistributionEndOp>(op);
        buildDistibution(begin, end, m, num_device);
        in_distribution = false;
        subf = std::make_shared<SubFunction>(0);
        insert_subop(subf, op);
      } else if (in_distribution) {
      } else if (subf == nullptr) {
        subf = std::make_shared<SubFunction>(0);
        insert_subop(subf, op);
      } else {
        insert_subop(subf, op);
      }
    }
  });
  if (subf != nullptr) {
    buildSubFunction(subf, m);
    subf = nullptr;
  }
  applyPattern<EndOpTranslate>(m);
}

} // namespace tpu
} // namespace tpu_mlir
