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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Patterns.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <llvm/ADT/iterator.h>
#include <set>
#include <sstream>

// TODO: move to mlir official impl in the next release
// example, walk<WalkOrder::PreOrder, ForwardDominanceIterator<true>>
WalkResult walk_pre(Operation *op, function_ref<WalkResult(Operation *)> callback) {
  WalkResult result = callback(op);
  if (result.wasSkipped())
    return WalkResult::advance();
  if (result.wasInterrupted())
    return WalkResult::interrupt();
  for (auto &region : op->getRegions()) {
    auto regionKindOp =
      dyn_cast_if_present<RegionKindInterface>(region.getParentOp());
    assert(regionKindOp && regionKindOp.hasSSADominance(region.getRegionNumber())
      && "graph regions are not allowed");
    Block *null = nullptr;
    auto it = llvm::make_pointee_range(region.empty()
                  ? llvm::make_range(llvm::df_end(null), llvm::df_end(null))
                  : llvm::depth_first(&region.front()));
    for (auto &block : it) {
      for (auto &nestedOp : block) {
        if (walk_pre(&nestedOp, callback).wasInterrupted())
          return WalkResult::interrupt();
      }
    }
  }
  return WalkResult::advance();
}

using namespace llvm;

using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {
void getInputsOutputs(std::vector<Operation *> &ops, std::vector<Value> &inputs,
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

class SubnetDividePass : public SubnetDivideBase<SubnetDividePass> {
public:
  SubnetDividePass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_REORDERED)) {
      llvm_unreachable("module should be reordered");
    }

    InfoVec subnet_infos = base_subnet_split();
    auto sorted_subnet_infos = sort_subnets(subnet_infos);
    sorted_subnet_infos = merge_sorted_subnets(sorted_subnet_infos);
    insert_merge_subnet(sorted_subnet_infos);
    reconstruct_ir(sorted_subnet_infos);
    for (auto info : sorted_subnet_infos)
      delete info;
    toposort();
    // for static ops
    auto &ctx = getContext();
    auto mOp = getOperation();
    for (auto func : mOp.getOps<FuncOp>()) {
      RewritePatternSet patterns(&ctx);
      // clang-format off
      if (getRunMode(func) == tpu::RunMode::TPU_STATIC) {
        patterns.add<patterns::ConvertPattern<tpu::UnsqueezeOp, tpu::ReshapeOp>,
                     patterns::ConvertPattern<tpu::SqueezeOp, tpu::ReshapeOp>>(&ctx);
      }
      patterns.add<patterns::FuseRepeatPattern<tpu::ReshapeOp>,
                   patterns::FuseSameOp>(&ctx);
      // clang-format on
      applyPatternsAndFoldGreedily(func, std::move(patterns));
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
    if (mode == RunMode::CPU || mode == RunMode::SWITCH)
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

  InfoVec base_subnet_split() {
    subnet_basic_info::reset_id();
    InfoVec subnet_infos;
    std::vector<Operation *> all_ops;
    llvm::DenseSet<Value> valid_values;
    bool noUse;

    toposort();
    WalkResult ret =
            walk_pre(
							module::getMainFuncOp(),
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
                          user) != subnet_infos[i]->ops.end())
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

    // remove the unused op
    for (auto op : to_move_ops)
      op->erase();
    return subnet_infos;
  }

  InfoVec sort_subnets(InfoVec &subnet_infos) {
    gen_subnet_data_depence(subnet_infos);

    InfoVec sorted_subnets;
    int latest_switch_index = 0;
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
        } else {
          subnet->index = sorted_subnets.size();
          if (subnet_have_terminator(subnet->ops)) {
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
              prevs[0]->type == RunMode::MERGE)) {
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
        if (subnets[i]->type == RunMode::SWITCH)
          break;
        auto &prev_subnets = subnet_prev_map[subnets[i]];
        if (prev_subnets[0]->type == RunMode::SWITCH)
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
      if (subnet->type == RunMode::SWITCH || subnet->type == RunMode::CPU) {
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
    auto it = std::find_if(subnets.begin(), subnets.end(),
                           [&](const subnet_basic_info *item) {
                             return item->type == RunMode::SWITCH;
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
      }
      ++it;
    }

    erase_yieldop(subnets);
    subnets.back()->next_index.assign({-1});
    return;
  }

  void reconstruct_ir(InfoVec &subnets) {
    for (auto &subnet : subnets) {
      std::vector<Type> argType;
      std::vector<Type> resType;
      std::vector<Value> fnInputs;
      std::vector<Value> fnOutputs;
      bool has_NoneOp = false;
      getInputsOutputs(subnet->ops, fnInputs, fnOutputs, has_NoneOp);
      std::vector<Location> argLoc;
      OpBuilder builder(module::getCtx());

      if (subnet->type == RunMode::MERGE) {
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
              isa<top::NoneOp>(
                  module::getOriValue(it.value()).getDefiningOp())) {
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
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createSubnetDividePass() {
  return std::make_unique<SubnetDividePass>();
}
} // namespace tpu
} // namespace tpu_mlir
