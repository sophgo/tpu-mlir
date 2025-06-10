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
#include "tpu_mlir/Support/CustomLayer.h"
#include "tpu_mlir/Support/Patterns.h"

using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

// Permute can convert to Reshape in some situations.
// For example:
// [4,3,28,1] => [4,3,1,28]
// [4,3,1,28] => [4,1,3,28]
struct TopPermuteToReshape : public OpRewriterPatternEx<PermuteOp> {
public:
  TopPermuteToReshape(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PermuteOp>(context, "TopPermuteToReshape") {}

  LogicalResult matchAndRewriteImpl(PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    // todo
    std::vector<int64_t> shape = module::getShape(op.getInput());
    int dim_size = shape.size();
    int start = 0, end = dim_size - 1;
    auto order = module::getI64Array(op.getOrder());
    while (start < dim_size && start == order->at(start)) {
      start++;
    }
    while (end > start && end == order->at(end)) {
      end--;
    }
    bool do_reshape = true;
    int64_t sum = 1;
    for (int index = start; index <= end; index++) {
      sum *= shape[index];
      if (shape[index] != 1 && sum != shape[index]) {
        do_reshape = false;
        break;
      }
    }
    if (do_reshape && order->size() == 2 && order->at(0) == 1 &&
        order->at(1) == 0 && op.getInput().getDefiningOp() != nullptr) {
      auto nonzeroOp = dyn_cast<tpu::NonZeroOp>(op.getInput().getDefiningOp());
      if (nonzeroOp && nonzeroOp.getOrder().str() == "RowMajor")
        do_reshape = false;
    }
    if (do_reshape == false) {
      return failure();
    }
    std::vector<Value> operands;
    operands.emplace_back(op.getInput());
    operands.emplace_back(module::getNoneOp(op));
    auto reshape_op = rewriter.replaceOpWithNewOp<tpu::ReshapeOp>(
        op, op.getResult().getType(), operands);
    for (auto next_op : reshape_op.getResult().getUsers()) {
      if (isa<tpu::ConcatOp>(next_op)) {
        auto concat_op = dyn_cast<tpu::ConcatOp>(next_op);
        concat_op->setAttr("only_merge", rewriter.getBoolAttr(false));
      }
    }
    return success();
  }
  bool shouldPrint(PermuteOp op) const override { return false; }
};

// slice + slice => slice
struct StaticMergeSlicePattern : public OpRewriterPatternEx<SliceOp> {
public:
  StaticMergeSlicePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SliceOp>(context, "StaticMergeSlicePattern") {}

  LogicalResult matchAndRewriteImpl(SliceOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isNone(op.getOffsetT()) || !module::isNone(op.getEndsT()) ||
        !module::isNone(op.getStepsT())) {
      return failure();
    }
    auto in_op = op.getInput().getDefiningOp();
    if (!in_op || !isa<SliceOp>(in_op) || in_op->hasOneUse() == false) {
      return failure();
    }
    auto output_shape = module::getShape(op.getOutput());
    auto num_dims = output_shape.size();
    auto in_slice = cast<SliceOp>(in_op);
    auto cur_offset = module::getI64Array(op.getOffset());
    auto cur_ends = module::getI64Array(op.getEnds());
    auto cur_steps = module::getI64Array(op.getSteps());
    auto in_offset = module::getI64Array(in_slice.getOffset());
    auto in_steps = module::getI64Array(in_slice.getSteps());

    std::vector<int64_t> new_offset(num_dims, 0);
    std::vector<int64_t> new_ends(num_dims, 0);
    std::vector<int64_t> new_steps(num_dims, 1);
    for (int i = 0; i < num_dims; i++) {
      auto cur_off = cur_offset->at(i);
      auto cur_end = cur_ends->at(i);
      auto cur_s = cur_steps->at(i);
      assert(cur_s > 0);
      auto in_off = in_offset->at(i);
      auto in_s = in_steps->at(i);
      assert(in_s > 0);
      new_offset[i] = in_off + cur_off * in_s;
      new_ends[i] = new_offset[i] + (cur_end - cur_off) * in_s;
      new_steps[i] = in_s * cur_s;
    }
    op->setAttr("offset", rewriter.getI64ArrayAttr(new_offset));
    op->setAttr("ends", rewriter.getI64ArrayAttr(new_ends));
    op->setAttr("steps", rewriter.getI64ArrayAttr(new_steps));
    op->setOperand(0, in_slice.getInput());
    rewriter.eraseOp(in_op);
    return success();
  }
  bool shouldPrint(SliceOp op) const override { return false; }
};

static void getInputsOutputs(std::vector<Operation *> &ops,
                             std::vector<Value> &inputs,
                             std::vector<Value> &outputs, bool &has_NoneOp) {
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
        has_NoneOp = true;
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

struct subnet_basic_info;
using InfoVec = llvm::SmallVector<subnet_basic_info *>;
struct subnet_basic_info {
  subnet_basic_info() : id(__next_id) {
    index = -1;
    __next_id++;
  }
  subnet_basic_info(int index_, std::vector<int> &&next_index_, RunMode type_,
                    const int id_ = 0)
      : index(index_), next_index(std::move(next_index_)), type(type_),
        id(id_) {
    __next_id++;
  }

  void clear_io() noexcept {
    ins.clear();
    outs.clear();
  }
  static void reset_id() { __next_id = 0; }
  RunMode type;
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
int subnet_basic_info::__next_id = 0;

static inline void LoopMode(Operation *op, int &mode) {
  if (isa<top::NoneOp>(op->getOperand(0).getDefiningOp())) {
    if (isa<top::WeightOp>(op->getOperand(1).getDefiningOp()) &&
        cast<top::WeightOp>(op->getOperand(1).getDefiningOp())
                .read_as_float()
                ->data()[0] == 1.0f) {
      mode = 3; // do_while
    } else
      mode = 4; // while
  }

  if (isa<top::NoneOp>(op->getOperand(1).getDefiningOp())) {
    mode = 5; // for
  }

  if (!isa<top::NoneOp>(op->getOperand(0).getDefiningOp()) &&
      !isa<top::NoneOp>(op->getOperand(0).getDefiningOp())) {
    /* input (trip_count, cond)
        int trip_count = ...;
        bool cond = ...;
        for (int i=0; i < trip_count && cond; ++i) {
            cond = ...;
        }
    */
    mode = 6;
  }

  if (isa<top::NoneOp>(op->getOperand(0).getDefiningOp()) &&
      isa<top::NoneOp>(op->getOperand(0).getDefiningOp())) {
    /* input (\"\", \"\"):
      for (int i=0; ; ++i) {
        cond = ... // Note this value is ignored, but is required in the body
      }
    */
    mode = 7; // loop forerver
    llvm_unreachable(
        "fatal error(loop forerver), please modify the origin model");
  }

  return;
}

/* note: if subnet's output order is the reverse order of module's,
   the final output's order is the reverse order of model. if static codegen,
   do L2S firstly, then S2S(to the target addr), so it have no error.
   but at dyn runtime, if it is the output tensor, it will L2S to the target
   addr according to the tensor id, it will meet error. so it will check if need
   to swap the order*/
class CallOpReorderPattern : public OpRewriterPatternEx<CallOp> {
public:
  CallOpReorderPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<CallOp>(context, "CallOpReorderPattern") {}
  static bool isOrderValid(const std::vector<int> &order) {
    for (int i = 0; i < order.size(); i++) {
      if (i != order[i]) {
        return true;
      }
    }
    return false;
  }

  static void reorderFuncOpInput(func::FuncOp func,
                                 const std::vector<int> &order) {
    // Get the function's input types and arguments
    auto &entryBlock = func.front();
    auto numArgs = func.getNumArguments();

    // Save the current input values
    std::vector<Value> originalArgs;
    for (auto &arg : entryBlock.getArguments()) {
      originalArgs.push_back(arg);
    }

    // Create a new function type with reordered input types
    SmallVector<Type, 4> reorderedInputTypes;
    SmallVector<Location, 4> reorderedLocs;
    for (auto i : order) {
      reorderedInputTypes.push_back(func.getFunctionType().getInput(i));
      reorderedLocs.push_back(entryBlock.getArgument(i).getLoc());
    }

    // Create the new function type with reordered inputs and the same outputs
    auto newFuncType = FunctionType::get(func.getContext(), reorderedInputTypes,
                                         func.getFunctionType().getResults());

    // Set the new function type
    func.setType(newFuncType);

    // Reorder the arguments in the entry block
    SmallVector<Value, 4> newArgs;
    for (int i = 0; i < numArgs; i++) {
      newArgs.push_back(
          entryBlock.addArgument(reorderedInputTypes[i], reorderedLocs[i]));
    }

    // Replace old arguments with new arguments in the function body
    for (int i = 0; i < numArgs; ++i) {
      originalArgs[order[i]].replaceAllUsesWith(newArgs[i]);
    }

    // Remove the old arguments
    for (int i = numArgs - 1; i >= 0; --i) {
      entryBlock.eraseArgument(i);
    }
  }

  static bool reorderCallOpInput(FuncOp main, CallOp call) {
    auto num_input = call.getNumOperands();
    if (main.getNumArguments() != num_input || num_input == 1) {
      return false;
    }
    std::vector<int> input_order(num_input);
    for (int i = 0; i < num_input; i++) {
      auto in = call.getOperand(i);
      if (!isa<top::InputOp>(in.getDefiningOp())) {
        return false;
      }
      auto in_op = cast<top::InputOp>(in.getDefiningOp());
      auto arg = cast<BlockArgument>(in_op.getInput());
      input_order[arg.getArgNumber()] = i;
    }
    if (!isOrderValid(input_order)) {
      return false;
    }

    SmallVector<Value, 4> inputs(call.getOperands());
    for (int i = 0; i < input_order.size(); i++) {
      call.setOperand(i, inputs[input_order[i]]);
    }
    auto m = module::getModuleOp(call);
    auto func = module::getFuncOp(m, call.getCallee());
    reorderFuncOpInput(func, input_order);
    return true;
  }

  static void reorderFuncOpOutput(func::FuncOp func,
                                  const std::vector<int> &order) {
    // Get the function's output types
    auto funcType = func.getFunctionType();

    // Create a new function type with reordered output types
    SmallVector<Type, 4> reorderedOutputTypes;
    for (int index : order) {
      reorderedOutputTypes.push_back(funcType.getResult(index));
    }

    // Create the new function type with reordered outputs and the same inputs
    auto newFuncType = FunctionType::get(
        func.getContext(), funcType.getInputs(), reorderedOutputTypes);

    // Set the new function type
    func.setType(newFuncType);

    // Assume the last operation in the function body is the return operation

    Operation &lastOp = func.getBody().front().back();
    auto returnOp = dyn_cast_or_null<func::ReturnOp>(lastOp);
    ASSERT_OP(returnOp, &lastOp);
    // Save the current return values
    SmallVector<Value, 4> originalResults(returnOp.getOperands());

    // Create a new list of return values based on the new order
    SmallVector<Value, 4> reorderedResults;
    for (int index : order) {
      reorderedResults.push_back(originalResults[index]);
    }

    // Set the new return operands
    returnOp->setOperands(reorderedResults);
  }

  static bool reorderCallOpOutput(FuncOp main, CallOp call) {
    auto num_output = call.getNumResults();
    if (main.getNumResults() != num_output || num_output == 1) {
      return false;
    }
    auto nextOp = call->getNextNode();
    auto retOp = dyn_cast_or_null<ReturnOp>(nextOp);
    if (!retOp || retOp.getNumOperands() != num_output) {
      return false;
    }
    std::vector<int> output_order;
    for (int i = 0; i < num_output; i++) {
      auto v = cast<OpResult>(retOp.getOperand(i));
      auto index = v.getResultNumber();
      if (v != call.getResult(index)) {
        return false;
      }
      output_order.push_back(index);
    }
    if (!isOrderValid(output_order)) {
      return false;
    }
    for (int i = 0; i < num_output; ++i) {
      retOp.setOperand(i, call.getResult(i));
    }
    auto m = module::getModuleOp(call);
    auto func = module::getFuncOp(m, call.getCallee());
    reorderFuncOpOutput(func, output_order);
    return true;
  }

  LogicalResult matchAndRewriteImpl(CallOp call,
                                    PatternRewriter &rewriter) const override {
    auto main = dyn_cast_or_null<FuncOp>(call->getParentOp());
    if (!main) {
      return failure();
    }
    auto do_input = reorderCallOpInput(main, call);
    auto do_output = reorderCallOpOutput(main, call);
    return (do_input || do_output) ? success() : failure();
  }
  bool shouldPrint(CallOp call) const override { return false; }
};

class SubnetDividePass : public SubnetDivideBase<SubnetDividePass> {
public:
  SubnetDividePass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_REORDERED)) {
      llvm_unreachable("module should be reordered");
    }
    auto &ctx = getContext();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      InfoVec subnet_infos = base_subnet_split(s);
      auto sorted_subnet_infos = sort_subnets(subnet_infos);
      sorted_subnet_infos = merge_sorted_subnets(sorted_subnet_infos);
      insert_merge_subnet(sorted_subnet_infos);
      reconstruct_ir(sorted_subnet_infos, s);
      for (auto info : sorted_subnet_infos)
        delete info;
      toposort();
      // for static ops
      for (auto func : s.getOps<FuncOp>()) {
        RewritePatternSet patterns(&ctx);
        if (getRunMode(func) == tpu::RunMode::TPU_STATIC) {
          patterns.add<TopPermuteToReshape, StaticMergeSlicePattern,
                       patterns::TPUUnsqueezeToReshapePattern,
                       patterns::TPUSqueezeToReshapePattern>(&ctx);
        }
        patterns.add<patterns::FuseRepeatPattern<tpu::ReshapeOp>,
                     patterns::FuseSameOp>(&ctx);
        applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
      module::applyPatternOnce<CallOpReorderPattern>(s);
    }

    module::removeUnusedOp();
    module::updateModuleTypes();
    module::setState(module::State::TPU_DIVIDED);
  }

  static bool force_dynamic_run(Operation *op) {
    if (isa<TopKOp, YoloDetectionOp, DetectionOutputOp, RoiAlignOp, NonZeroOp,
            NmsOp, SortOp>(op)) {
      return true;
    } else if (isa<MaskRCNNRPNGetBboxesOp, MaskRCNNBboxPoolerOp,
                   MaskRCNNGetBboxBOp, MaskRCNNMaskPoolerOp>(op)) {
      return true;
    } else if (isa<RoiExtractorOp>(op)) {
      auto RoiExtractorOp = dyn_cast<tpu::RoiExtractorOp>(op);
      if (RoiExtractorOp.getIsStatic()) {
        return false;
      } else {
        return true;
      }
    } else if (op->hasTrait<trait::ShapeProducer>()) {
      return true;
    } else if (op->hasTrait<trait::ShapeConsumer>()) {
      return true;
    } else if (isa<SliceOp>(op)) {
      return !module::isNone(dyn_cast<SliceOp>(op).getOffsetT());
    } else if (isa<InterpOp>(op)) {
      return !module::isNone(dyn_cast<InterpOp>(op).getShapeT());
    } else if (module::isBM1684Family()) {
      if (isa<tpu::GatherOp>(op)) {
        return true;
      }
    } else if (isa<tpu::CustomOp>(op)) {
      auto custom_op = dyn_cast<tpu::CustomOp>(op);
      auto params = custom_op.getParams();
      std::vector<custom_param_t> values;
      values.push_back({0});
      customOpProcessParam(params, values);
      std::string op_name = custom_op.getName().str();
      std::string api_name = "force_dynamic_run_" + op_name;
      bool ret = false;
      BM168x::call_custom_plugin_func(
          kCustomPluginTypes::PLUGIN_FORCEDYNAMICRUN, &ret, api_name.c_str(),
          values.data(), values.size() * sizeof(custom_param_t), nullptr);
      return ret;
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
    } else if (isa<tpu::LoopOp>(op)) {
      seperate = true;
      return RunMode::LOOP;
    } else if (dynamic || force_dynamic_run(op)) {
      if (module::isMARS3()) {
        UNREACHABLE_THIS("Arch not support dynamic subnet, please do constant "
                         "folding and use static mode");
      }
      return RunMode::TPU_DYNAMIC;
    } else if (isa<CustomOp>(op)) {
      if (dyn_cast<CustomOp>(op).getName().starts_with("ap")) {
        return RunMode::CPU;
      }
    }
    return RunMode::TPU_STATIC;
  }

  bool toposortAction(Block *block, llvm::iterator_range<Block::iterator> ops) {
    auto isOpReady = [&](Operation *op,
                         llvm::DenseSet<Operation *> &unscheduledOps) -> bool {
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

  bool op_can_run(Operation *op, llvm::DenseSet<Value> &valid_values) {
    int valid_num = 0;
    for (int i = 0; i < op->getNumOperands(); i++) {
      Operation *parent = op->getOperand(i).getDefiningOp();
      if (!parent) {
        valid_num++;
        continue;
      }
      if (isa<top::WeightOp, top::NoneOp>(parent) ||
          valid_values.count(op->getOperand(i)))
        valid_num++;
    }
    return valid_num == op->getNumOperands();
  }

  bool is_special_op(Operation *op) {
    bool nouse;
    auto mode = getOpMode(op, nouse);
    if (mode == RunMode::CPU || mode == RunMode::SWITCH ||
        mode == RunMode::LOOP)
      return true;
    else
      return false;
  }

  void erase_yieldop(InfoVec &subnets) {
    // erase the yieldOp for to rebuild the ir
    for (auto &subnet : subnets) {
      if (subnet_have_terminator(subnet->ops)) {
        auto &ops = subnet->ops;
        auto it = std::find_if(ops.begin(), ops.end(), [&](Operation *op) {
          return isa<tpu::YieldOp>(op);
        });
        ops.erase(it);
      }
    }
    return;
  }

  bool subnet_have_terminator(std::vector<Operation *> ops) {
    for (auto &op : ops) {
      if (isa<tpu::YieldOp>(op))
        return true;
    }
    return false;
  }

  bool subnet_can_run(subnet_basic_info *subnet, const InfoVec &run_group) {
    auto valid_count = subnet->prev_subnets.size();
    for (auto &in : subnet->prev_subnets) {
      if (std::find(run_group.begin(), run_group.end(), in) ==
          std::end(run_group))
        valid_count--;
    }

    return valid_count == subnet->prev_subnets.size();
  }

  void gen_subnet_data_depence(InfoVec &subnet_infos) {
    llvm::DenseMap<Value, subnet_basic_info *> value_from_subnets;
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

  void update_subnet_io(subnet_basic_info *subnet) {
    subnet->clear_io();

    for (auto it = subnet->ops.begin(); it != subnet->ops.end(); ++it) {
      if (!isa<top::WeightOp, top::NoneOp>(*it)) {
        for (int k = 0; k < (*it)->getNumResults(); k++) {
          for (auto user : (*it)->getResult(k).getUsers()) {
            if (std::find(subnet->ops.begin(), subnet->ops.end(), user) ==
                subnet->ops.end())
              subnet->outs.emplace_back((*it)->getResult(k));
          }
        }

        auto op = *it;
        for (int k = 0; k < op->getNumOperands(); k++) {
          if (!op->getOperand(k).isa<BlockArgument>() &&
              std::find(subnet->ops.begin(), subnet->ops.end(),
                        op->getOperand(k).getDefiningOp()) ==
                  subnet->ops.end()) {
            subnet->ins.emplace_back(op->getOperand(k));
          }
        }
      }
    }
    return;
  }

  bool belong_to_if_subnet(subnet_basic_info *subnet) {
    bool ret = false;
    for (int i = 0; i < subnet->ops.size(); i++) {
      if (isa<tpu::YieldOp>(subnet->ops[i]) &&
          isa<tpu::IfOp>(subnet->ops[i]->getParentOp()))
        ret = true;
    }
    return ret;
  }

  bool belong_to_loop_subnet(subnet_basic_info *subnet) {
    bool ret = false;
    for (int i = 0; i < subnet->ops.size(); i++) {
      if (isa<tpu::YieldOp>(subnet->ops[i]) &&
          isa<tpu::LoopOp>(subnet->ops[i]->getParentOp()))
        ret = true;
    }
    return ret;
  }

  Value insert_host2device(Value v, Type to) {
    auto ctx = v.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(v);
    auto name = module::getName(v).str();
    name += "_host2device";
    auto newType =
        RankedTensorType::get(module::getShape(v), module::getStorageType(v));
    auto loc = NameLoc::get(builder.getStringAttr(name));
    auto hdOp = builder.create<tpu::Host2DeviceOp>(loc, newType, ValueRange{v});
    return hdOp.getOutput();
  }

  Value insert_device2host(Value v, Type to, Operation *user) {
    auto ctx = v.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(v);
    auto name = module::getName(v).str();
    if (user && !isa<ReturnOp>(user)) {
      name += "_" + module::getName(user).str();
    }
    name += "_device2host";
    auto newType =
        RankedTensorType::get(module::getShape(v), module::getStorageType(v));
    auto loc = NameLoc::get(builder.getStringAttr(name));
    auto hdOp = builder.create<tpu::Device2HostOp>(loc, newType, ValueRange{v});
    return hdOp.getOutput();
  }

  InfoVec base_subnet_split(ModuleOp sub) {
    subnet_basic_info::reset_id();
    InfoVec subnet_infos;
    std::vector<Operation *> all_ops;
    llvm::DenseSet<Value> valid_values;
    bool noUse;
    auto main = module::getMainFuncOp(sub);
    toposort();
    main.walk<WalkOrder::PreOrder, ForwardDominanceIterator<true>>(
        [&](Operation *op) {
          if (isa<func::FuncOp, ReturnOp, top::NoneOp>(op))
            return WalkResult::advance();
          else if (isa<top::InputOp>(op)) {
            valid_values.insert(op->getResult(0));
            return WalkResult::advance();
          } else if (isa<top::WeightOp>(op)) {
            valid_values.insert(op->getResult(0));
          }
          all_ops.emplace_back(op);
          return WalkResult::advance();
        });

    bool has_switch_op = false;
    bool has_loop_op = false;
    auto dfs = [&]() noexcept -> void {
      while (!all_ops.empty()) {
        subnet_infos.emplace_back(new subnet_basic_info);
        auto &info = subnet_infos.back();
        info->type = dynamic ? RunMode::TPU_DYNAMIC : RunMode::TPU_STATIC;
        bool updated = false;
        do {
          updated = false;
          for (auto op : all_ops) {
            if (op_can_run(op, valid_values)) {
              if (is_special_op(op)) {
                if (info->ops.empty()) {
                  auto mode = getOpMode(op, noUse);
                  info->type = mode;
                  info->ops.emplace_back(op);
                  if (isa<tpu::IfOp>(op)) {
                    has_switch_op = true;
                  } else if (isa<tpu::LoopOp>(op)) {
                    has_loop_op = true;
                  } else {
                    valid_values.insert(op->result_begin(), op->result_end());
                  }
                  break;
                }
                continue;
              } else if (info->type != RunMode::TPU_DYNAMIC &&
                         force_dynamic_run(op)) {
                if (!info->ops.empty())
                  continue;
                info->type = RunMode::TPU_DYNAMIC;
                info->ops.emplace_back(op);
                valid_values.insert(op->result_begin(), op->result_end());
                if (isa<tpu::SortOp, tpu::TopKOp, tpu::GatherOp>(op)) {
                  updated = false;
                  break;
                } else {
                  updated = true;
                }
              } else {
                if (has_switch_op &&
                    (info->ops.empty() || info->ops[0]->getParentRegion() ==
                                              op->getParentRegion())) {
                  info->ops.emplace_back(op);
                  valid_values.insert(op->result_begin(), op->result_end());
                  if (isa<tpu::YieldOp>(op) &&
                      isa<tpu::IfOp>(op->getBlock()->getParentOp())) {
                    if (op->getParentRegion()->getRegionNumber()) {
                      valid_values.insert(
                          op->getBlock()->getParentOp()->result_begin(),
                          op->getBlock()->getParentOp()->result_end());
                      has_switch_op = false;
                    }
                    updated = false;
                    break;
                  } else {
                    updated = true;
                  }
                } else if (has_loop_op && (info->ops.empty() ||
                                           info->ops[0]->getParentRegion() ==
                                               op->getParentRegion())) {
                  info->ops.emplace_back(op);
                  valid_values.insert(op->result_begin(), op->result_end());
                  if (isa<tpu::YieldOp>(op) &&
                      isa<tpu::LoopOp>(op->getBlock()->getParentOp())) {
                    valid_values.insert(
                        op->getBlock()->getParentOp()->result_begin(),
                        op->getBlock()->getParentOp()->result_end());
                    has_loop_op = false;
                    updated = false;
                    break;
                  } else {
                    updated = true;
                  }
                } else if (isa<func::FuncOp>(op->getParentOp())) {
                  info->ops.emplace_back(op);
                  valid_values.insert(op->result_begin(), op->result_end());
                  updated = true;
                }
              }
            }
          }

          for (auto op : info->ops) {
            auto it = std::find(all_ops.begin(), all_ops.end(), op);
            if (it != all_ops.end())
              all_ops.erase(it);
          }
        } while (updated);
      }
    };

    /* use dfs to traverse the SSA, build the subgraph
       according to data depence , control flow, Op type */
    dfs();

    /*  1. move the WeightOp and NoneOp's position between subnets
           and get the input and output of subnet

        2. if subnet's output in host -> host2device  and  device2host (in next
       subnet)*/

    std::vector<Operation *> to_move_ops;
    for (int i = 0; i < subnet_infos.size(); i++) {
      bool add_h2d_flag = false;
      Value h2dval, d2hval;
      std::vector<Operation *> h2dvec;
      std::vector<std::pair<Operation *, int>> d2hvec;
      for (auto it = subnet_infos[i]->ops.begin();
           it != subnet_infos[i]->ops.end();) {
        if (!isa<top::WeightOp, top::NoneOp>(*it)) {
          for (int k = 0; k < (*it)->getNumResults(); k++) {
            for (auto user : (*it)->getResult(k).getUsers()) {
              if (std::find(subnet_infos[i]->ops.begin(),
                            subnet_infos[i]->ops.end(),
                            user) == subnet_infos[i]->ops.end()) {
                // if subnet's output in host -> host2device
                auto output_op_p = (*it);
                if ((output_op_p)->hasTrait<trait::ShapeProducer>()) {
                  // if Op hasTrait<trait::ShapeProducer>()), Op is in host !
                  for (auto user : output_op_p->getResult(k).getUsers()) {
                    // note: user is in another subnet
                    for (auto idx = 0; idx < user->getNumOperands(); idx++) {
                      // insert host2device
                      if (user->getOperand(idx) == output_op_p->getResult(k)) {

                        d2hval = insert_device2host(
                            user->getOperand(idx),
                            user->getOperand(idx).getType(), user);
                        user->setOperand(idx, d2hval);
                        auto d2hop = d2hval.getDefiningOp();

                        h2dval =
                            insert_host2device(d2hop->getOperand(0),
                                               d2hop->getOperand(0).getType());
                        d2hop->setOperand(0, h2dval);
                        // find which subnet_info that d2hval add in
                        for (int l = 0; l < subnet_infos.size(); l++) {
                          if (std::find(subnet_infos[l]->ops.begin(),
                                        subnet_infos[l]->ops.end(),
                                        user) != subnet_infos[l]->ops.end()) {
                            d2hvec.push_back({d2hval.getDefiningOp(), l});
                          }
                        }
                        break;
                      }
                    }
                  }
                  add_h2d_flag = true; // subnet_infos[i] can't updata now!
                  h2dvec.emplace_back(h2dval.getDefiningOp());
                  // h2dvec.emplace_back(d2hval.getDefiningOp());
                  subnet_infos[i]->outs.emplace_back(
                      h2dval); // h2dval is subnet output now
                  break;
                } else {
                  // subnet's output is in device,
                  subnet_infos[i]->outs.emplace_back((*it)->getResult(k));
                  break;
                }
              }
            }
          }

          auto op = *it;
          for (int k = 0; k < op->getNumOperands(); k++) {
            if (!op->getOperand(k).isa<BlockArgument>() &&
                std::find(subnet_infos[i]->ops.begin(),
                          subnet_infos[i]->ops.end(),
                          op->getOperand(k).getDefiningOp()) ==
                    subnet_infos[i]->ops.end()) {
              auto iit = std::find(to_move_ops.begin(), to_move_ops.end(),
                                   op->getOperand(k).getDefiningOp());
              if (iit != to_move_ops.end()) {
                it = subnet_infos[i]->ops.insert(
                    it + 1, op->getOperand(k).getDefiningOp());
                to_move_ops.erase(iit);
              } else
                subnet_infos[i]->ins.emplace_back(op->getOperand(k));
            }
          }
          ++it;
        } else {
          bool move_flag = true;
          for (auto user : (*it)->getResult(0).getUsers()) {
            if (std::find(subnet_infos[i]->ops.begin(),
                          subnet_infos[i]->ops.end(),
                          user) != subnet_infos[i]->ops.end() ||
                isa<tpu::LoopOp>(user))
              /* if user is loopOp, also don't
                 move the weightOp and NoneOp */
              move_flag = false;
          }
          if (move_flag) {
            to_move_ops.emplace_back(*it);
            it = subnet_infos[i]->ops.erase(it);
          } else {
            ++it;
          }
        }
      }

      // if need to add host2device Op and device2host Op, updata now !
      if (add_h2d_flag) {
        // add host2device Op
        for (auto h2dOp : h2dvec) {
          subnet_infos[i]->ops.emplace_back(h2dOp);
        }
        // add device2host Op
        for (auto d2hOp : d2hvec) {
          subnet_infos[d2hOp.second]->ops.emplace_back(d2hOp.first);
        }
      }
    }

    // eliminate empty subnet
    for (auto it = subnet_infos.begin(); it < subnet_infos.end();) {
      if ((*it)->ops.empty()) {
        it = subnet_infos.erase(it);
      } else {
        ++it;
      }
    }

    // if have loopOp, create some ir & subnets
    for (auto it = subnet_infos.begin(); it != subnet_infos.end();) {
      if ((*it)->type == RunMode::LOOP) {
        // check loop mode
        int mode = 0;
        LoopMode((*it)->ops[0], mode);
        OpBuilder builder(module::getCtx());
        OpBuilder::InsertionGuard insertGuard(builder);
        if (mode == 3) {

        } else if (mode == 4) {

        } else if (mode == 5) {

        } else if (mode == 6) {
          builder.setInsertionPoint((*it)->ops[0]);
          auto value2 = (*it)->ops[0]->getOperand(0);
          std::vector<NamedAttribute> attrs;
          double init_float = 0;
          double step_v = 1.0f;
          std::vector<Operation *> ops;
          /* if Loop's Operand (such as cond, v_initial) is WeightOp,
             will create tensor for to replace it, because will change
            the value during iteration */
          for (int i = 0; i < (*it)->ops[0]->getNumOperands() - 1; i++) {
            if (isa<top::WeightOp>(
                    module::getOriValue((*it)->ops[0]->getOperand(i + 1))
                        .getDefiningOp())) {
              // create a D2D op(but use Addconst instead)
              attrs.clear();
              attrs.push_back(builder.getNamedAttr(
                  "const_val", builder.getF64FloatAttr(0.f)));
              auto d2d_loc =
                  module::getLocLike((*it)->ops[0], std::to_string(i));
              auto d2d_op = builder.create<tpu::D2DOp>(
                  d2d_loc, (*it)->ops[0]->getOperand(i + 1).getType(),
                  ValueRange{(*it)->ops[0]->getOperand(i + 1)}, attrs);
              ops.emplace_back(d2d_op);
              (*it)->ops[0]->getOperand(i + 1).replaceUsesWithIf(
                  d2d_op->getResult(0), [&](OpOperand &use) {
                    return isa<tpu::LoopOp>(use.getOwner());
                  });
            }
          }

          // create WeightOp
          auto init_v = std::make_shared<std::vector<float>>(1);
          init_v->data()[0] = 1;
          auto shape = value2.getType().template cast<ShapedType>().getShape();
          auto type = RankedTensorType::get(
              shape, value2.getType().cast<ShapedType>().getElementType());
          auto init_Value =
              top::WeightOp::create((*it)->ops[0], "_init", *init_v, type);
          // create constfill op
          auto fill_loc = module::getLocLike((*it)->ops[0], "_fill");
          attrs.clear();
          attrs.push_back(builder.getNamedAttr(
              "value", builder.getF64FloatAttr(init_float)));
          auto fill = builder.create<tpu::ConstantFillOp>(
              fill_loc, value2.getType(), ValueRange{init_Value}, attrs);

          // create CompareOp
          auto cmp_loc = module::getLocLike((*it)->ops[0], "_less");
          attrs.clear();
          attrs.push_back(
              builder.getNamedAttr("mode", builder.getStringAttr("Less")));
          auto cmp = builder.create<tpu::CompareOp>(
              cmp_loc, value2.getType(), ValueRange{fill->getResult(0), value2},
              attrs);

          // create AutoIncreaseOp
          auto autoincrease_loc =
              module::getLocLike((*it)->ops[0], "_AutoIncrease");
          attrs.clear();
          attrs.push_back(builder.getNamedAttr(
              "const_val", builder.getF64FloatAttr(step_v)));
          auto new_autoincrease = builder.create<tpu::AutoIncreaseOp>(
              autoincrease_loc, value2.getType(),
              ValueRange{fill->getResult(0)}, attrs);

          // create and Op
          auto and_loc = module::getLocLike((*it)->ops[0], "_and");
          attrs.clear();
          attrs.push_back(
              builder.getNamedAttr("mode", builder.getStringAttr("And")));
          auto and_op = builder.create<tpu::CompareOp>(
              and_loc, cmp->getResult(0).getType(),
              ValueRange{cmp->getResult(0), (*it)->ops[0]->getOperand(1)},
              attrs);
          /* insert autoIncrease(for to not remove unused op)
            & and_op's output into LoopOp's operand */
          (*it)->ops[0]->insertOperands(
              (*it)->ops[0]->getNumOperands(),
              {new_autoincrease->getResult(0), and_op->getResult(0)});
          if (it != subnet_infos.begin()) {
            // insert WeightOp & constFill op into previous subnet
            auto iit = std::prev(it, 1);
            if (!ops.empty()) {
              (*iit)->ops.insert((*iit)->ops.end(), ops.begin(), ops.end());
            }

            (*iit)->ops.insert((*iit)->ops.end(),
                               {init_Value.getDefiningOp(), fill});
            update_subnet_io(*iit);
          } else {
            // insert a new subnet
            it = subnet_infos.insert(subnet_infos.begin(),
                                     new subnet_basic_info);
            (*it)->type = dynamic ? RunMode::TPU_DYNAMIC : RunMode::TPU_STATIC;
            if (!ops.empty()) {
              (*it)->ops.insert((*it)->ops.end(), ops.begin(), ops.end());
            }

            (*it)->ops.insert((*it)->ops.end(),
                              {init_Value.getDefiningOp(), fill});
            update_subnet_io(*it);
            it = std::next(it, 1);
          }

          // insert a loop entring or head subnet
          it = subnet_infos.insert(it, new subnet_basic_info);
          (*it)->type = dynamic ? RunMode::TPU_DYNAMIC : RunMode::TPU_STATIC;
          (*it)->ops.insert((*it)->ops.end(), {cmp, new_autoincrease, and_op});
          update_subnet_io(*it);
          std::advance(it, 2);
        }
      } else {
        ++it;
      }
    }
    // remove the unused op
    for (auto op : to_move_ops) {
      if (!op->getUsers().empty()) {
        continue;
      }
      op->erase();
    }

    return subnet_infos;
  }

  InfoVec sort_subnets(InfoVec &subnet_infos) {
    gen_subnet_data_depence(subnet_infos);

    InfoVec sorted_subnets;
    int latest_switch_index = 0;
    int latest_prehead_index = 0;
    for (auto &subnet : subnet_infos) {
      if (subnet_can_run(subnet, sorted_subnets)) {
        if (subnet->type == RunMode::SWITCH) {
          subnet->index = sorted_subnets.size();
          subnet->next_index.push_back(subnet->index + 1);
          auto it = std::find(subnet_infos.begin(), subnet_infos.end(), subnet);
          latest_switch_index = std::distance(subnet_infos.begin(), it);
          for (auto iit = std::next(it, 1); iit < subnet_infos.end(); ++iit) {
            if (subnet_have_terminator((*iit)->ops) && (*iit)->outs.empty() &&
                (*iit)->next_subnets.empty()) {
              subnet->next_index.push_back(
                  std::distance(subnet_infos.begin(), iit + 1));
              break;
            }
          }
          sorted_subnets.emplace_back(subnet);
        } else if (subnet->type == RunMode::LOOP) {
          subnet->index = sorted_subnets.size();
          subnet->next_index.push_back(subnet->index + 1); // loop body
          auto it = std::find(subnet_infos.begin(), subnet_infos.end(), subnet);
          latest_prehead_index =
              std::distance(subnet_infos.begin(), it - 1); // prehead
          for (auto iit = std::next(it, 1); iit < subnet_infos.end(); ++iit) {
            if (subnet_have_terminator((*iit)->ops) && (*iit)->outs.empty() &&
                (*iit)->next_subnets.empty()) {
              subnet->next_index.push_back(
                  std::distance(subnet_infos.begin(), iit + 1));
              break;
            }
          }
          sorted_subnets.emplace_back(subnet);
        } else {
          subnet->index = sorted_subnets.size();
          if (subnet_have_terminator(subnet->ops) &&
              belong_to_if_subnet(subnet)) {
            // find the merge position
            auto it =
                std::find(subnet_infos.begin(), subnet_infos.end(), subnet);
            for (auto iit = std::next(it, 1); iit < subnet_infos.end(); ++iit) {
              auto &next_subnets =
                  subnet_infos[latest_switch_index]->next_subnets;
              if (!next_subnets.empty() &&
                  std::find(next_subnets.begin(), next_subnets.end(), *iit) !=
                      next_subnets.end()) {
                subnet->next_index.push_back(
                    std::distance(subnet_infos.begin(), iit));
                break;
              }
            }
          } else if (subnet_have_terminator(subnet->ops) &&
                     belong_to_loop_subnet(subnet)) {
            // set the prehead position
            subnet->next_index.push_back(latest_prehead_index);
          } else {
            subnet->next_index.push_back(subnet->index + 1);
          }
          sorted_subnets.emplace_back(subnet);
        }
      } else {
        // can;t run to here, if happen, pls let us know ASAP
        llvm_unreachable("Fatal error, pls let us know ASAP.");
      }
    }

    sorted_subnets.back()->next_index.clear();
    return sorted_subnets;
  }

  bool has_special_dyn_op_static_shape(subnet_basic_info *&subnet,
                                       InfoVec &subnets) {
    auto it = std::find_if(
        subnet->ops.begin(), subnet->ops.end(), [&](const Operation *op) {
          return isa<tpu::SortOp, tpu::TopKOp, tpu::GatherOp>(op);
        });
    if (it == subnet->ops.end())
      return false;
    // check the type of previous subnet of current dynamic subnet
    auto iit = std::find(subnets.begin(), subnets.end(), subnet);
    if (iit == subnets.begin())
      return false;
    else {
      int index = std::distance(subnets.begin(), iit);
      auto previous_static_subnet = [&](auto &&Me, InfoVec &subnets,
                                        int current_index) noexcept -> InfoVec {
        InfoVec prevs;
        std::copy_if(std::begin(subnets), std::end(subnets),
                     std::back_inserter(prevs),
                     [&](subnet_basic_info *&subnet) {
                       auto gg = std::find_if(
                           subnet->next_index.begin(), subnet->next_index.end(),
                           [&](int vv) { return vv == current_index; });
                       return gg != subnet->next_index.end();
                     });

        if (!(prevs[0]->type == RunMode::SWITCH ||
              prevs[0]->type == RunMode::MERGE ||
              prevs[0]->type == RunMode::LOOP)) {
          return prevs;
        } else {
          auto kk = std::find(subnets.begin(), subnets.end(), prevs[0]);
          if (kk == subnets.begin())
            return prevs;
          int index_ = std::distance(subnets.begin(), kk);
          return Me(Me, subnets, index_);
        }
      };

      /* the previous subnet of dynamic subnet is
        static, don't transfer dynamic to succeccors,
        if previous subnet is switch/merge, need to check
        the previous subnet of switch/merge continuously */
      auto pre_subnets =
          previous_static_subnet(previous_static_subnet, subnets, index);
      if (pre_subnets.size() == 1 &&
          pre_subnets[0]->type == RunMode::TPU_STATIC)
        return true;
      else if (pre_subnets.size() >= 2) {
        /* merge subnet have at least two previous subnet,
         as long as there is one dynamic, can transfer
         dynamic to successors */
        for (int i = 0; i < pre_subnets.size(); i++) {
          if (pre_subnets[i]->type == RunMode::TPU_DYNAMIC)
            return false;
        }
        return true;
      } else
        return false;
    }
  }

  InfoVec merge_sorted_subnets(InfoVec &subnets) {
    int size = subnets.size();
    std::map<subnet_basic_info *, int> subnet_prev_count;
    std::map<subnet_basic_info *, std::vector<subnet_basic_info *>>
        subnet_prev_map;
    for (auto &subnet : subnets) {
      for (auto next : subnet->next_index) {
        subnet_prev_count[subnets[next]]++;
        subnet_prev_map[subnets[next]].emplace_back(subnet);
      }
    }

    // broadcast the subnet type to successors
    for (int i = 1; i < size; i++) {
      for (int j = 0; j < subnet_prev_count[subnets[i]]; j++) {
        if (subnets[i]->type == RunMode::SWITCH ||
            subnets[i]->type == RunMode::LOOP)
          break;
        auto &prev_subnets = subnet_prev_map[subnets[i]];
        if (prev_subnets[0]->type == RunMode::SWITCH ||
            prev_subnets[0]->type == RunMode::LOOP)
          break;
        if (subnets[i]->type == RunMode::TPU_STATIC &&
            prev_subnets[j]->type == RunMode::TPU_DYNAMIC) {
          /* dynamic subnet which has GatherOp/Topk op
             and the previous of this dynamic subnet has static shape,
             then don;t transfer the dynamic to successors */
          if (!has_special_dyn_op_static_shape(prev_subnets[j], subnets)) {
            subnets[i]->type = RunMode::TPU_DYNAMIC;
          }
          break;
        }
      }
    }

    for (int i = 0; i < size; i++) {
      auto subnet = subnets[i];
      if (subnet == nullptr)
        continue;
      if (subnet->type == RunMode::SWITCH || subnet->type == RunMode::CPU ||
          subnet->type == RunMode::LOOP) {
        continue;
      }

      for (int j = i + 1; j < size; j++) {
        auto next_subnet = subnets[j];
        if (!next_subnet)
          continue;
        if (next_subnet->type == subnet->type && !subnet->next_index.empty() &&
            subnet->next_index[0] == j && subnet_prev_count[next_subnet] == 1) {
          subnet->ops.insert(subnet->ops.end(), next_subnet->ops.begin(),
                             next_subnet->ops.end());
          subnet->ins.clear();
          subnet->outs.clear();
          subnet->next_index = next_subnet->next_index;
          next_subnet->ops.clear();
          delete next_subnet;
          subnets[j] = nullptr;
          // no need to update the ins/outs
        }
      }
    }

    InfoVec merged_subnets;
    std::map<subnet_basic_info *, int> merged_index;
    for (auto subnet : subnets) {
      if (subnet) {
        subnet->index = merged_subnets.size();
        merged_index[subnet] = subnet->index;
        merged_subnets.emplace_back(subnet);
      }
    }

    for (auto subnet : merged_subnets) {
      for (auto &index : subnet->next_index) {
        if (index < size) {
          index = merged_index[subnets[index]];
        } else {
          index = -1;
        }
      }
    }

    return merged_subnets;
  }

  void insert_merge_subnet(InfoVec &subnets) {
    auto it = std::find_if(
        subnets.begin(), subnets.end(), [&](const subnet_basic_info *item) {
          return (item->type == RunMode::SWITCH || item->type == RunMode::LOOP);
        });
    if (it == subnets.end()) {
      subnets.back()->next_index.assign({-1});
      erase_yieldop(subnets);
      return;
    }

    for (auto it = subnets.begin(); it < subnets.end();) {
      if ((*it)->type == RunMode::SWITCH) {
        // insert merge subnet to nearest position
        for (auto iit = it + 1; iit < subnets.end(); iit++) {
          if (subnet_have_terminator((*iit)->ops)) {
            if (!(*iit)->next_index.empty()) {
              int next_merge_pos = (*iit)->next_index[0];
              auto merged_subnet = subnets.insert(
                  subnets.begin() + next_merge_pos,
                  new subnet_basic_info(next_merge_pos, {next_merge_pos + 1},
                                        RunMode::MERGE));
              std::copy((*it)->outs.begin(), (*it)->outs.end(),
                        std::back_inserter((*merged_subnet)->ins));
              for (int i = next_merge_pos + 1; i < subnets.size(); i++) {
                subnets[i]->index++;
                if (!subnets[i]->next_index.empty())
                  std::for_each(subnets[i]->next_index.begin(),
                                subnets[i]->next_index.end(),
                                [&](int &v) { v++; });
              }
            } else {
              int next_merge_pos = subnets.size();
              auto merged_subnet = subnets.insert(
                  subnets.begin() + next_merge_pos,
                  new subnet_basic_info(next_merge_pos, {-1}, RunMode::MERGE));
              std::copy((*it)->outs.begin(), (*it)->outs.end(),
                        std::back_inserter((*merged_subnet)->ins));
              for (auto kk = iit; kk < subnets.end(); kk++) {
                if (subnet_have_terminator((*kk)->ops)) {
                  (*kk)->next_index.assign({next_merge_pos});
                }
              }
            }
            break;
          }
        }
      } else if ((*it)->type == RunMode::LOOP) {
        // insert two merge subnets to exiting(exiting->prehead: backedge) and
        // exit postion
        for (auto iit = it + 1; iit < subnets.end(); iit++) {
          if (subnet_have_terminator((*iit)->ops)) {
            // 1. the exiting in the loop-body
            int insert_pos = std::distance(subnets.begin(), iit) + 1;
            int next_merge_pos = (*iit)->next_index[0];
            auto merged_subnet = subnets.insert(
                subnets.begin() + insert_pos,
                new subnet_basic_info(insert_pos, {next_merge_pos},
                                      RunMode::MERGE));
            // special handle
            auto yield_it = std::find_if(
                (*iit)->ops.begin(), (*iit)->ops.end(),
                [&](Operation *op) { return isa<tpu::YieldOp>(op); });
            for (int k = 0; k < (*yield_it)->getNumOperands(); k++)
              (*merged_subnet)->ins.emplace_back((*yield_it)->getOperand(k));

            std::vector<int>{insert_pos}.swap((*iit)->next_index);
            (*it)->next_index[1]++;
            for (int i = insert_pos + 1; i < subnets.size(); i++) {
              subnets[i]->index++;
              if (!subnets[i]->next_index.empty())
                std::for_each(subnets[i]->next_index.begin(),
                              subnets[i]->next_index.end(),
                              [&](int &v) { v++; });
            }
          } else if ((*it)->next_index[1] ==
                     std::distance(subnets.begin(), iit)) {
            int insert_pos = std::distance(subnets.begin(), iit);
            // 1. exit postion of the loop
            auto merged_subnet = subnets.insert(
                iit, new subnet_basic_info(insert_pos, {insert_pos + 1},
                                           RunMode::MERGE));
            std::copy((*it)->outs.begin(), (*it)->outs.end(),
                      std::back_inserter((*merged_subnet)->ins));
            for (int i = insert_pos + 1; i < subnets.size(); i++) {
              subnets[i]->index++;
              if (!subnets[i]->next_index.empty())
                std::for_each(subnets[i]->next_index.begin(),
                              subnets[i]->next_index.end(),
                              [&](int &v) { v++; });
            }
            break;
          }
        }
      }
      ++it;
    }

    erase_yieldop(subnets);
    subnets.back()->next_index.assign({-1});
    return;
  }

  void reconstruct_ir(InfoVec &subnets, ModuleOp submodule) {
    for (auto &subnet : subnets) {
      std::vector<Type> argType;
      std::vector<Type> resType;
      std::vector<Value> fnInputs;
      std::vector<Value> fnOutputs;
      bool has_NoneOp = false;
      getInputsOutputs(subnet->ops, fnInputs, fnOutputs, has_NoneOp);
      std::vector<Location> argLoc;
      OpBuilder builder(module::getCtx());
      OpBuilder::InsertionGuard insertGuard(builder);

      // for if's merge subnet and loop's exit merge subnet
      if (subnet->type == RunMode::MERGE &&
          !isa<BlockArgument>(subnet->ins[0]) &&
          isa<tpu::IfOp, tpu::LoopOp>(subnet->ins[0].getDefiningOp())) {
        auto funcOp =
            cast<FuncOp>(subnet->ins[0].getDefiningOp()->getParentOp());
        func::CallOp callee = module::getCallOp(funcOp);
        for (auto &&result : callee.getResults()) {
          fnInputs.emplace_back(result);
        }

        builder.setInsertionPointAfter(callee.getOperation());
        std::vector<Location> locs;
        std::vector<Type> outType;
        for (auto &v : fnInputs)
          outType.emplace_back(v.getType());

        for (int i = 0; i < fnInputs.size(); i++) {
          auto loc = module::getLocLike(module::getOriValue(fnInputs[i]),
                                        "_id_" + std::to_string(subnet->index));
          locs.push_back(loc);
        }

        auto new_loc = FusedLoc::get(module::getCtx(), locs);
        auto identityOp =
            builder.create<tpu::IdentityOp>(new_loc, outType, fnInputs);
        subnet->ops.emplace_back(identityOp.getOperation());
        for (auto &&v : identityOp.getOperation()->getResults()) {
          fnOutputs.emplace_back(v);
        }

        for (auto it : llvm::enumerate(callee.getResults())) {
          fnInputs[it.index()].replaceAllUsesExcept(fnOutputs[it.index()],
                                                    identityOp.getOperation());
        }
      } else if (subnet->type == RunMode::MERGE) {
        // the exiting/latch pos at loop-body
        std::vector<Value>().swap(fnInputs);
        std::copy(subnet->ins.begin(), subnet->ins.end(),
                  std::back_inserter(fnInputs));
        func::CallOp callee;
        for (int i = 0; i < fnInputs.size(); i++) {
          if (!isa<BlockArgument>(fnInputs[i])) {
            auto funcOp =
                cast<FuncOp>(fnInputs[i].getDefiningOp()->getParentOp());
            int index = 0;
            callee = module::getCallOp(funcOp);
            for (int k = 0; k < callee.getResults().size(); k++) {
              if (fnInputs[i] == funcOp.front().back().getOperand(k)) {
                index = k;
                break;
              }
            }

            fnInputs[i] = callee.getResult(index);
          }
        }
        // find the insert pos
        Block *pos;
        if (!isa<BlockArgument>(fnInputs[0])) {
          pos = fnInputs[0].getDefiningOp()->getBlock();
        } else {
          pos = cast<BlockArgument>(fnInputs[0]).getOwner();
        }

        builder.setInsertionPoint(pos, Block::iterator(pos->back()));

        std::vector<Location> locs;
        std::vector<Type> outType;
        for (auto &v : fnInputs)
          outType.emplace_back(v.getType());

        for (int i = 0; i < fnInputs.size(); i++) {
          auto loc = module::getLocLike(module::getOriValue(fnInputs[i]),
                                        "_id_" + std::to_string(subnet->index));
          locs.push_back(loc);
        }

        auto new_loc = FusedLoc::get(module::getCtx(), locs);
        auto identityOp =
            builder.create<tpu::IdentityOp>(new_loc, outType, fnInputs);
        subnet->ops.emplace_back(identityOp.getOperation());
        for (auto &&v : identityOp.getOperation()->getResults()) {
          fnOutputs.emplace_back(v);
        }

        for (int i = 0; i < identityOp.getOperation()->getNumResults(); i++) {
          pos->back().getOperand(i).replaceAllUsesExcept(
              identityOp.getOperation()->getResult(i),
              identityOp.getOperation());
        }
      }

      for (auto &input : fnInputs) {
        argType.push_back(input.getType());
        auto ori_input = module::getOriValue(input);
        if (!module::isNone(ori_input)) {
          argLoc.push_back(module::getLoc(ori_input));
        } else {
          argLoc.push_back(module::getLoc());
        }
      }

      for (auto &output : fnOutputs) {
        resType.push_back(output.getType());
      }
      int64_t id = subnet->index;
      std::string func_name = "subfunc_" + std::to_string(id);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          builder.getNamedAttr("id", builder.getI64IntegerAttr(id)));
      attrs.push_back(builder.getNamedAttr(
          "mode", RunModeAttr::get(module::getCtx(), subnet->type)));
      attrs.push_back(builder.getNamedAttr(
          "next_index", builder.getDenseI32ArrayAttr(subnet->next_index)));

      auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{argType},
                                            llvm::ArrayRef<Type>{resType});
      auto fnOp = FuncOp::create(module::getLoc(), func_name, fnType,
                                 ArrayRef<NamedAttribute>(attrs));
      auto block = fnOp.addEntryBlock();
      top::NoneOp noneOp;
      if (has_NoneOp) {
        builder.setInsertionPointToStart(block);
        noneOp = builder.create<top::NoneOp>(module::getLoc(),
                                             builder.getNoneType());
      }

      builder.setInsertionPoint(subnet->ops.back());

      func::CallOp callOp = builder.create<func::CallOp>(
          module::getLoc(), func_name, resType, fnInputs);
      for (auto it : llvm::enumerate(callOp.getResults())) {
        fnOutputs[it.index()].replaceUsesWithIf(
            it.value(), [&](OpOperand &operand) {
              Operation *user = operand.getOwner();
              return find(subnet->ops.begin(), subnet->ops.end(), user) ==
                     subnet->ops.end();
            });
      }

      builder.setInsertionPointToEnd(block);

      auto retOp = builder.create<ReturnOp>(module::getLoc(), fnOutputs);
      for (auto &op : subnet->ops) {
        for (auto it : llvm::enumerate(op->getOperands())) {
          if (!it.value().isa<BlockArgument>() &&
              isa<top::NoneOp>(it.value().getDefiningOp())) {
            op->setOperand(it.index(), noneOp);
          }
        }

        op->moveBefore(retOp);
      }

      submodule.push_back(fnOp);
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
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass() {
  return std::make_unique<SubnetDividePass>();
}
} // namespace tpu
} // namespace tpu_mlir
