//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Patterns.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Iterators.h"

using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {
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
          patterns
              .add<patterns::ConvertPattern<tpu::UnsqueezeOp, tpu::ReshapeOp>,
                   patterns::ConvertPattern<tpu::SqueezeOp, tpu::ReshapeOp>>(
                  &ctx);
        }
        patterns.add<patterns::FuseRepeatPattern<tpu::ReshapeOp>,
                     patterns::FuseSameOp>(&ctx);
        applyPatternsAndFoldGreedily(func, std::move(patterns));
      }
    }
    module::removeUnusedOp();
    module::setState(module::State::TPU_DIVIDED);
  }

  static bool force_dynamic_run(Operation *op) {
    if (isa<TopKOp, YoloDetectionOp, DetectionOutputOp, RoiAlignOp, NonZeroOp,
            NmsOp>(op)) {
      return true;
    } else if (op->hasTrait<trait::ShapeProducer>()) {
      return true;
    } else if (op->hasTrait<trait::ShapeConsumer>()) {
      return true;
    } else if (isa<SliceOp>(op)) {
      return !module::isNone(dyn_cast<SliceOp>(op).getOffsetT());
    } else if (module::isBM1684Family()) {
      if (auto gather_op = dyn_cast<tpu::GatherOp>(op)) {
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
    } else if (isa<tpu::LoopOp>(op)) {
      seperate = true;
      return RunMode::LOOP;
    } else if (dynamic || force_dynamic_run(op)) {
      return RunMode::TPU_DYNAMIC;
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
                if (isa<tpu::TopKOp, tpu::GatherOp>(op)) {
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

    /* move the WeightOp and NoneOp's position between subnets
       and get the input and output of subnet */
    std::vector<Operation *> to_move_ops;
    for (int i = 0; i < subnet_infos.size(); i++) {
      for (auto it = subnet_infos[i]->ops.begin();
           it != subnet_infos[i]->ops.end();) {
        if (!isa<top::WeightOp, top::NoneOp>(*it)) {
          for (int k = 0; k < (*it)->getNumResults(); k++) {
            for (auto user : (*it)->getResult(k).getUsers()) {
              if (std::find(subnet_infos[i]->ops.begin(),
                            subnet_infos[i]->ops.end(),
                            user) == subnet_infos[i]->ops.end())
                subnet_infos[i]->outs.emplace_back((*it)->getResult(k));
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
            (*iit)->ops.insert((*iit)->ops.end(),
                               {init_Value.getDefiningOp(), fill});
            update_subnet_io(*iit);
          } else {
            // insert a new subnet
            it = subnet_infos.insert(subnet_infos.begin(),
                                     new subnet_basic_info);
            (*it)->type = dynamic ? RunMode::TPU_DYNAMIC : RunMode::TPU_STATIC;
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
    for (auto op : to_move_ops)
      op->erase();

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
    auto it = std::find_if(subnet->ops.begin(), subnet->ops.end(),
                           [&](const Operation *op) {
                             return isa<tpu::TopKOp, tpu::GatherOp>(op);
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
        auto new_name =
            module::getName(module::getOriValue(fnInputs[0]).getDefiningOp())
                .str() +
            "_id_" + std::to_string(subnet->index);
        auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
        std::vector<Type> outType;
        for (auto &v : fnInputs)
          outType.emplace_back(v.getType());
        auto identityOp =
            builder.create<tpu::IdentityOp>(name_loc, outType, fnInputs);
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
        auto new_name =
            module::getName(module::getOriValue(fnInputs[0]).getDefiningOp())
                .str() +
            "_id_" + std::to_string(subnet->index);
        auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
        std::vector<Type> outType;
        for (auto &v : fnInputs)
          outType.emplace_back(v.getType());
        auto identityOp =
            builder.create<tpu::IdentityOp>(name_loc, outType, fnInputs);
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
