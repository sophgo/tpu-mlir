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
  SubFunction(int64_t devid, int64_t step) : devid(devid), step(step) {
    need_none = false;
  }
  int64_t devid;
  int64_t step;
  bool need_none;
  std::vector<Operation *> ops;
};

static constexpr llvm::StringRef DEVICE_ID = "device_id";
static constexpr llvm::StringRef STEP = "step";

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

static void buildSubFunction(std::shared_ptr<SubFunction> sf, ModuleOp m) {
  if (sf == nullptr || sf->ops.empty()) {
    return;
  }
  auto ctx = m.getContext();
  std::vector<Value> fnInputs;
  std::vector<Value> fnOutputs;
  getInputsOutputs(sf->ops, fnInputs, fnOutputs);
  std::vector<Type> argType;
  std::vector<Type> resType;
  std::vector<Location> argLoc;
  std::vector<Location> retLoc;
  for (auto input : fnInputs) {
    argType.push_back(input.getType());
    argLoc.push_back(module::getLoc(input));
  }
  for (auto output : fnOutputs) {
    resType.push_back(output.getType());
    retLoc.push_back(module::getLoc(output));
  }
  auto name = module::getName(m).str();
  std::string func_name =
      name + "_" + std::to_string(sf->step) + "_" + std::to_string(sf->devid);
  OpBuilder builder(m.getContext());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr(STEP, builder.getI64IntegerAttr(sf->step)));
  attrs.push_back(
      builder.getNamedAttr(DEVICE_ID, builder.getI64IntegerAttr(sf->devid)));
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{argType},
                                        llvm::ArrayRef<Type>{resType});
  auto fnOp =
      FuncOp::create(NameLoc::get(builder.getStringAttr(func_name)), func_name,
                     fnType, ArrayRef<NamedAttribute>(attrs));
  auto block = fnOp.addEntryBlock();
  builder.setInsertionPointAfterValue(fnOutputs.back());
  Location call_loc = retLoc[0];
  if (retLoc.size() > 1) {
    call_loc = FusedLoc::get(ctx, retLoc);
  }
  func::CallOp callOp =
      builder.create<func::CallOp>(call_loc, func_name, resType, fnInputs);
  for (auto it : llvm::enumerate(callOp.getResults())) {
    fnOutputs[it.index()].replaceUsesWithIf(
        it.value(), [&](OpOperand &operand) {
          Operation *user = operand.getOwner();
          return find(sf->ops.begin(), sf->ops.end(), user) == sf->ops.end();
        });
  }
  builder.setInsertionPointToStart(block);
  top::NoneOp noneOp;
  if (sf->need_none) {
    noneOp = builder.create<top::NoneOp>(m.getLoc(), builder.getNoneType());
  }
  auto retOp = builder.create<func::ReturnOp>(call_loc, fnOutputs);
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
  m.push_back(fnOp);
  if (sf->need_none) {
    builder.setInsertionPointAfter(noneOp);
  } else {
    builder.setInsertionPointToStart(block);
  }
  for (auto it : llvm::enumerate(fnInputs)) {
    auto v = it.value();
    auto idx = it.index();
    auto arg = block->getArgument(idx);
    arg.setLoc(argLoc[idx]);
    auto input =
        builder.create<top::InputOp>(argLoc[idx], v.getType(), ValueRange{arg});
    v.replaceUsesWithIf(input, [&](OpOperand &operand) {
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
    } else if (isa<top::NoneOp>(op_) && subf->need_none == false) {
      subf->need_none = true;
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
    } else if (isa<top::NoneOp>(op_) && subf->need_none == false) {
      subf->need_none = true;
    }
  }
  subf->ops.push_back(op);
}

// MatMulTopK use forward
static void collect_ops_forward(std::shared_ptr<SubFunction> &subf,
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
    collect_ops_forward(subf, u);
  }
}

static void buildDistibution(tpu::DistributionBeginOp begin,
                             tpu::DistributionEndOp end, ModuleOp m,
                             int64_t num_devices, int64_t step) {
  std::vector<Operation *> begins(begin->user_begin(), begin->user_end());
  std::vector<Value> ends(end->operand_begin(), end->operand_end());
  for (int i = 0; i < num_devices; i++) {
    auto subf = std::make_shared<SubFunction>(i, step);
    if (begins.size() == num_devices) {
      collect_ops_forward(subf, begins[i]);
    } else if (ends.size() == num_devices) {
      collect_ops_backward(subf, ends[i].getDefiningOp());
    } else {
      llvm_unreachable("Not Implemented");
    }
    buildSubFunction(subf, m);
  }
}

static std::shared_ptr<SubFunction> buildEndOp(tpu::DistributionEndOp end,
                                               ModuleOp m, int64_t num_devices,
                                               int64_t &step) {
  OpBuilder builder(end.getContext());
  builder.setInsertionPointAfter(end);
  int times = num_devices > 2 ? std::ceil(std::sqrt(num_devices)) : 1;
  std::vector<Value> operands(end.operand_begin(), end.operand_end());
  std::vector<Value> new_operands;
  std::vector<std::shared_ptr<tpu_mlir::tpu::SubFunction>> subf_v;
  auto mode = getEndMode(end);
  switch (mode) {
  case DistributionEndMode::EndToSum:
    for (int t = 0; t < times; t++) {
      for (int i = 0; i < operands.size(); i += 2) {
        if (i + 1 == operands.size()) {
          new_operands.push_back(operands[i]);
          continue;
        }
        auto loc = module::getLocLike(operands[i],
                                      std::string("add_") + std::to_string(i));
        auto add = builder.create<tpu::AddOp>(
            loc, operands[i].getType(),
            mlir::ValueRange{operands[i], operands[i + 1]});
        new_operands.push_back(add.getOutput());
        auto subf = std::make_shared<SubFunction>(
            i * (int)std::pow(2, t) % num_devices, step);
        insert_subop(subf, add);
        if (t == times - 1) {
          module::setLoc(add.getOutput(), module::getLoc(end.getOutput()));
          end.getOutput().replaceAllUsesWith(add.getOutput());
          end.erase();
          for (auto f : subf_v) {
            buildSubFunction(f, m);
          }
          return std::move(subf);
        } else {
          subf_v.emplace_back(std::move(subf));
        }
      }
      step++;
      operands = new_operands;
      new_operands.clear();
    }
    break;
  case DistributionEndMode::EndToTopK:
    for (int t = 0; t < times; t++) {
      for (int i = 0; i < operands.size(); i += 4) {
        if (i + 2 == operands.size()) {
          new_operands.push_back(operands[i]);
          new_operands.push_back(operands[i + 1]);
          continue;
        }
        auto value = operands[i];
        auto indice = operands[i + 1];
        auto value2 = operands[i + 2];
        auto indice2 = operands[i + 3];
        auto loc =
            module::getLocLike(operands[i + 1], "cmp_" + std::to_string(i));
        std::vector<NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            "mode", builder.getStringAttr("GreaterOrEqual")));
        auto cmp = builder.create<tpu::CompareOp>(
            loc, value.getType(), ValueRange{value, value2}, attrs);
        loc = module::getLocLike(operands[i], "value_" + std::to_string(i));
        auto value_select = builder.create<tpu::WhereOp>(
            loc, value.getType(), ValueRange{cmp.getOutput(), value, value2});
        value = value_select.getOutput();
        loc =
            module::getLocLike(operands[i + 1], "indice_" + std::to_string(i));
        auto indice_select = builder.create<tpu::WhereOp>(
            loc, indice.getType(),
            ValueRange{cmp.getOutput(), indice, indice2});
        indice = indice_select.getOutput();
        new_operands.push_back(value);
        new_operands.push_back(indice);
        auto subf = std::make_shared<SubFunction>(
            (i / 2 * (int)std::pow(2, t)) % num_devices, step);
        insert_subop(subf, cmp);
        insert_subop(subf, indice_select);
        if (t == times - 1) {
          module::setLoc(indice, module::getLoc(end.getOutput()));
          end.getOutput().replaceAllUsesWith(indice);
          end.erase();
          for (auto f : subf_v) {
            buildSubFunction(f, m);
          }
          return std::move(subf);
        } else {
          insert_subop(subf, value_select);
          subf_v.emplace_back(std::move(subf));
        }
      }
      step++;
      operands = new_operands;
      new_operands.clear();
    }
    break;
  default:
    llvm_unreachable("Not Implemented");
    break;
  }
  return nullptr;
}

static int64_t getDeviceId(FuncOp func) {
  return func->getAttrOfType<IntegerAttr>(DEVICE_ID).getInt();
}

static int64_t getStep(FuncOp func) {
  return func->getAttrOfType<IntegerAttr>(STEP).getInt();
}

class Function2Module : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    auto m = module::getModuleOp();
    auto func = module::getFuncOp(m, op.getCallee());
    auto device_id = getDeviceId(func);
    auto step = getStep(func);
    rewriter.setInsertionPointToStart(m.getBody());
    auto sub_m = rewriter.create<ModuleOp>(func.getLoc(), op.getCallee());
    module::setSubModuleId(sub_m, device_id, step);
    func->removeAttr(DEVICE_ID);
    func->removeAttr(STEP);
    func->moveBefore(sub_m.getBody(), sub_m.getBody()->begin());
    func.setName("main");
    return success();
  }
};

static void distributeToOneModule(ModuleOp m) {
  auto func = module::getFuncOp(m, "main");
  OpBuilder builder(m.getContext());
  builder.setInsertionPointToStart(m.getBody());
  auto sub_m = builder.create<ModuleOp>(m.getLoc(), module::getName(m));
  module::setSubModuleId(sub_m, 0, 0);
  func->moveBefore(sub_m.getBody(), sub_m.getBody()->begin());
}

void distributeModules(ModuleOp m, int64_t num_device) {
  auto main = module::getMainFuncOp(m);
  std::vector<StringRef> input_names;
  std::vector<StringRef> output_names;
  for (auto in : main.getOps<top::InputOp>()) {
    input_names.push_back(module::getName(in.getOutput()));
  }
  for (auto ret : main.getOps<func::ReturnOp>()) {
    for (auto v : ret.getOperands()) {
      output_names.push_back(module::getName(v));
    }
  }
  module::setInputs(input_names);
  module::setOutputs(output_names);
  if (num_device == 1) {
    distributeToOneModule(m);
    return;
  }

  std::shared_ptr<SubFunction> subf = nullptr;
  bool in_distribution = false;
  int64_t step = 0;
  tpu::DistributionBeginOp begin;
  // split to different functions
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
        buildDistibution(begin, end, m, num_device, step++);
        in_distribution = false;
        subf = buildEndOp(end, m, num_device, step);
      } else if (in_distribution) {
        // do nothing
      } else if (subf == nullptr) {
        subf = std::make_shared<SubFunction>(0, step++);
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
  // each function create one module
  applyPatternOnce<Function2Module>(m);
  // remove main, and functions
  auto ops = m.getOps<FuncOp>();
  std::vector<FuncOp> funcs(ops.begin(), ops.end());
  for (auto f : funcs) {
    f.erase();
  }
  // make moudle order to be step by step and device by device
  auto subs = m.getOps<ModuleOp>();
  std::vector<ModuleOp> modules(subs.begin(), subs.end());
  if (modules.size() <= 1) {
    return;
  }
  std::sort(modules.begin(), modules.end(), [](ModuleOp a, ModuleOp b) {
    int64_t a_devid, a_step, b_devid, b_step;
    module::getSubModuleId(a, a_devid, a_step);
    module::getSubModuleId(b, b_devid, b_step);
    return a_step < b_step || (a_step == b_step && a_devid < b_devid);
  });
  for (int i = 1; i < modules.size(); i++) {
    modules[i]->moveAfter(modules[i - 1]);
  }
}

} // namespace tpu
} // namespace tpu_mlir
