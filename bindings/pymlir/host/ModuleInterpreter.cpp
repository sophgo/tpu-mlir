//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ModuleInterpreter.h"
#include "progressbar.hpp"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/GmemAllocator.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <algorithm>
#define DEBUG_TYPE "interpreter"

static const int64_t MAX_COUNT_LIMIT = 0x100000000ll;
namespace tpu_mlir {
using namespace tpu;
ModuleInterpreter::ModuleInterpreter(ModuleOp module) : module(module) {
  module::init(module);
  if (!module::isState(module::State::TOP_F32) &&
      !module::isState(module::State::TPU_LOWERED)) {
    llvm_unreachable("mlir state not support");
  }
  mem_mode = mem_mode_t::ALL_TENSOR_IN_MEM;
  total_count = 0;
  for (auto func : module.getOps<FuncOp>()) {
    // alloce buffer for all value
    func.walk([&](InferenceInterface op) {
      for (auto r : op->getResults()) {
        total_count += module::getNumElements(r);
      }
    });
  }
  LLVM_DEBUG(llvm::dbgs() << "Allocate size: "
                          << total_count * sizeof(float) / 1024 << " KB\n");
  if (total_count >= MAX_COUNT_LIMIT) {
    // if not work, try rebuild with mem_mode_t::PART_SMALL_TENSOR_IN_MEM
    mem_mode = mem_mode_t::PART_TENSOR_IN_MEM;
  }
}

ModuleInterpreter::~ModuleInterpreter() {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = module::getName(op).str();
        if (inference_map.find(name) != inference_map.end()) {
          infer_op.deinit(*inference_map[name]);
        }
      }
    });
  }
}

bool ModuleInterpreter::is_no_mem_op(Operation *op) {
  if (op == nullptr) {
    return false;
  }
  return isa<top::ReshapeOp, tpu::ReshapeOp, top::UnsqueezeOp, top::SqueezeOp>(
      op);
}

void ModuleInterpreter::allocate_resources() {
  switch (mem_mode) {
  case mem_mode_t::ALL_TENSOR_IN_MEM:
    allocate_all_tensor_in_mem();
    break;
  case mem_mode_t::PART_TENSOR_IN_MEM:
    allocate_part_tensor_in_mem();
    break;
  case mem_mode_t::ALL_TENSOR_IN_DISK:
    allocate_all_tensor_in_disk();
    break;
  case mem_mode_t::PART_SMALL_TENSOR_IN_MEM:
    allocate_small_tensor_in_mem();
    break;
  case mem_mode_t::ALL_TENSOR_IN_REUSED_MEM:
    allocate_tensor_in_reused_mem();
    break;
  }
}

void ModuleInterpreter::allocate_tensor_in_reused_mem() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto name = module::getName(v).str();
          output_names.push_back(name);
        }
      } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto v = in_op.getOutput();
        auto name = module::getName(v).str();
        input_names.push_back(name);
        value_map[name] = v;
        all_tensor_names.push_back(name);
      } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
        auto v = wOp.getOutput();
        auto name = module::getName(v).str();
        value_map[name] = v;
        mem_map[name] = wOp.read_as_float();
        all_weight_names.push_back(name);
      } else {
        for (auto v : op->getResults()) {
          if (module::getNumElements(v) == 0)
            continue;
          auto name = module::getName(v).str();
          all_tensor_names.push_back(name);
          value_map[name] = v;
        }
      }
    });
  }

  module::detachWeightFile(); // free weight mem

  uint32_t loc = 0;
  int64_t alignment = 0;
  int64_t start_addr = 0;
  std::map<ValueInfo, TensorLive> liveRange;
  std::map<Operation *, uint32_t> ops_loc;
  std::vector<ValueInfo> common_ops;
  // std::vector<ValueInfo> inplace_ops;
  std::vector<Operation *> infer_ops;
  std::map<ValueInfo, int64_t> gaddrMap;
  std::map<ValueInfo, std::string> vNameMap;
  uint64_t gmemUsed = 0;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (llvm::dyn_cast<InferenceInterface>(op)) {
        num_infer_op++;
        ops_loc[op] = loc;
        loc++;
        infer_ops.emplace_back(op);
      } else if (llvm::dyn_cast<top::InputOp>(op) || isa<ReturnOp>(op)) {
        ops_loc[op] = loc;
        loc++;
        infer_ops.emplace_back(op);
      }
    });
  }
  for (Operation *infer_op : infer_ops) {
    if (!isa<ReturnOp>(infer_op)) {
      // int num_results = infer_op->getNumResults();
      int index = 0;
      for (auto v : infer_op->getResults()) {
        // auto iter = infer_ops.end();
        if (!module::isNone(v)) {
          ValueInfo v_info(infer_op, index);
          TensorLive v_live(ops_loc[infer_op], ops_loc[infer_op],
                            module::getNumElements(v));
          common_ops.emplace_back(v_info);
          liveRange[v_info] = v_live;

          for (auto successor_op : v.getUsers()) {
            if (ops_loc.count(successor_op) > 0) {
              liveRange[v_info].end =
                  (liveRange[v_info].end > ops_loc[successor_op])
                      ? liveRange[v_info].end
                      : ops_loc[successor_op];
            }
          }
          liveRange[v_info].end++;
          vNameMap[v_info] = module::getName(v).str();
        }
        index++;
      }
    }
  }
  if (!common_ops.empty()) {
    GmemAllocator::sortOpByLiveStart(common_ops, liveRange);
    GmemAllocator allocator(gaddrMap, alignment);
    gmemUsed = allocator.assignGaddr(common_ops, liveRange, true, start_addr);
    std::cout << "reused mem is " << gmemUsed << ", all mem is " << total_count
              << std::endl;
  }
  auto gMem = std::make_shared<std::vector<float>>(gmemUsed);
  // float* ptr = gMem->data();
  for (auto pair : gaddrMap) {
    auto name = vNameMap[pair.first];
    auto memLoc = pair.second;
    auto size = liveRange[pair.first].tensor_size;
    mem_map[name] = gMem;
    activation_offset[name] = std::make_pair(memLoc, size);
    // std::cout << name << " start_addr " << ptr << " cur_addr " << (float
    // *)(mem_map[name]->data()) - ptr << std::endl;
  }
  for (Operation *op : infer_ops) {
    if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
      std::string op_name = module::getName(op).str();
      auto param = std::make_shared<InferenceParameter>();
      for (auto result : op->getResults()) {
        if (result.getType().isa<NoneType>())
          param->outputs.push_back(nullptr);
        else {
          auto r_name = module::getName(result).str();
          auto r_memLoc = activation_offset[r_name].first;
          param->outputs.push_back(mem_map[r_name]->data() + r_memLoc);
        }
      }
      for (auto input : op->getOperands()) {
        if (module::isNone(input)) {
          param->inputs.push_back(nullptr);
          continue;
        }
        auto input_name = module::getName(input).str();

        if (mem_map.find(input_name) == mem_map.end()) {
          input.dump();
          llvm_unreachable("input operands not allocated");
        } else {
          auto input_memLoc = activation_offset[input_name].first;
          param->inputs.push_back(mem_map[input_name]->data() + input_memLoc);
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "init: '" << op_name << "'\n");
      if (failed(infer_op.init(*param))) {
        op->dump();
        llvm_unreachable("op inferece init failed");
      }
      inference_map[op_name] = param;
    }
  }
}

bool ModuleInterpreter::check_op_in_mem(Operation *op) {
  for (auto r : op->getResults()) {
    if (module::isNone(r)) {
      continue;
    } else {
      auto name = module::getName(r).str();
      if (mem_map.find(name) == mem_map.end() ||
          mem_map[name].use_count() == 0) {
        return false;
      }
    }
  }
  for (auto i : op->getOperands()) {
    if (module::isNone(i)) {
      continue;
    } else {
      auto name = module::getName(i).str();
      if (mem_map.find(name) == mem_map.end() ||
          mem_map[name].use_count() == 0) {
        return false;
      }
    }
  }
  return true;
}

void ModuleInterpreter::collect_tensor(Value v) {
  auto count = module::getNumElements(v);
  if (count == 0) {
    return;
  }
  auto name = module::getName(v).str();
  if (value_map.find(name) != value_map.end()) {
    return;
  }
  mem_map[name] = std::make_shared<std::vector<float>>(count);
  value_map[name] = v;
  all_tensor_names.push_back(name);
}

void ModuleInterpreter::allocate_part_tensor_in_mem() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  int step = ceiling_func(total_count, MAX_COUNT_LIMIT);
  int64_t idx = 0;
  for (auto func : module.getOps<FuncOp>()) {
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto name = module::getName(v).str();
          output_names.push_back(name);
          collect_tensor(v);
          if (auto castOp = dyn_cast<tpu::CastOp>(v.getDefiningOp())) {
            collect_tensor(castOp.getOutput());
          }
        }
      } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto v = in_op.getOutput();
        collect_tensor(v);
        auto name = module::getName(v).str();
        input_names.push_back(name);
      } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
        auto v = wOp.getOutput();
        auto name = module::getName(v).str();
        value_map[name] = v;
        mem_map[name] = wOp.read_as_float();
        all_weight_names.push_back(name);
      } else {
        for (auto r : op->getResults()) {
          auto num_users = std::distance(r.user_begin(), r.user_end());
          if (num_users > 1) {
            collect_tensor(r);
          } else if (idx % (2 * step) < step) {
            collect_tensor(r);
          }
        }
        idx++;
      }
    });
    module::detachWeightFile(); // to free weight memory
    // input output buffers for ops
    func.walk([&](InferenceInterface infer_op) {
      num_infer_op++;
      auto name = module::getName(infer_op).str();
      // checkout in and out in memory
      if (check_op_in_mem(infer_op)) {
        auto param = std::make_shared<InferenceParameter>();
        for (auto result : infer_op->getResults()) {
          if (result.getType().isa<NoneType>()) {
            param->outputs.push_back(nullptr);
          } else {
            auto o_name = module::getName(result).str();
            param->outputs.push_back(mem_map[o_name]->data());
          }
        }
        for (auto input : infer_op->getOperands()) {
          if (module::isNone(input)) {
            param->inputs.push_back(nullptr);
          } else {
            auto i_name = module::getName(input).str();
            param->inputs.push_back(mem_map[i_name]->data());
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
        if (failed(infer_op.init(*param))) {
          infer_op->dump();
          llvm_unreachable("op inferece init failed");
        }
        inference_map[name] = param;
      }
    });
  }
}
void ModuleInterpreter::allocate_all_tensor_in_disk() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  for (auto func : module.getOps<FuncOp>()) {
    // only weight, input and output save in memory
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto name = module::getName(v).str();
          output_names.push_back(name);
          // only output in ddr. other tensors in disk
          auto count = module::getNumElements(v);
          mem_map[name] = std::make_shared<std::vector<float>>(count);
          all_tensor_names.push_back(name);
        }
      } else {
        for (auto result : op->getResults()) {
          auto count = module::getNumElements(result);
          if (count == 0) {
            continue;
          }
          auto name = module::getName(result).str();
          bool is_input = isa<top::InputOp>(op);
          if (is_input) {
            input_names.push_back(name);
          }
          value_map[name] = result;
          if (auto wOp = llvm::dyn_cast<top::WeightOp>(op)) {
            mem_map[name] = wOp.read_as_float();
            all_weight_names.push_back(name);
          } else if (is_input) {
            mem_map[name] = std::make_shared<std::vector<float>>(count);
            all_tensor_names.push_back(name);
          }
        }
      }
    });
    module::detachWeightFile(); // to free weight memory
  }
}
void ModuleInterpreter::allocate_all_tensor_in_mem() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  for (auto func : module.getOps<FuncOp>()) {
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          collect_tensor(v);
          auto name = module::getName(v).str();
          output_names.push_back(name);
        }
      } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto v = in_op.getOutput();
        collect_tensor(v);
        auto name = module::getName(v).str();
        input_names.push_back(name);
      } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
        auto v = wOp.getOutput();
        auto name = module::getName(v).str();
        mem_map[name] = wOp.read_as_float();
        all_weight_names.push_back(name);
        value_map[name] = v;
      } else if (is_no_mem_op(op)) {
        auto v = op->getResult(0);
        auto name = module::getName(v).str();
        auto in = module::getName(op->getOperand(0)).str();
        mem_map[name] = mem_map[in];
        all_tensor_names.push_back(name);
        value_map[name] = v;
      } else {
        for (auto r : op->getResults()) {
          collect_tensor(r);
        }
      }
    });
    module::detachWeightFile(); // to free weight memory

    // input output buffers for all ops
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        num_infer_op++;
        auto name = module::getName(op).str();
        auto param = std::make_shared<InferenceParameter>();
        for (auto result : op->getResults()) {
          if (result.getType().isa<NoneType>()) {
            param->outputs.push_back(nullptr);
          } else if (0 == module::getNumElements(result)) {
            param->outputs.push_back(nullptr);
          } else {
            auto o_name = module::getName(result).str();
            param->outputs.push_back(mem_map[o_name]->data());
          }
        }
        for (auto input : op->getOperands()) {
          std::string input_name;
          if (module::isNone(input)) {
            param->inputs.push_back(nullptr);
            continue;
          } else if (input.isa<BlockArgument>()) {
            /* op support nested ops,
               can transfer operands by blockargument */
            std::size_t index = input.cast<BlockArgument>().getArgNumber();
            Value vv = input.cast<BlockArgument>()
                           .getOwner()
                           ->getParentOp()
                           ->getOperands()[index];
            input_name = module::getName(vv).str();
          } else {
            input_name = module::getName(input).str();
          }

          if (mem_map.find(input_name) == mem_map.end()) {
            if (module::getNumElements(input) == 0) {
              param->inputs.push_back(nullptr);
            } else {
              input.dump();
              llvm_unreachable("input operands not allocated");
            }
          } else {
            param->inputs.push_back(mem_map[input_name]->data());
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
        if (failed(infer_op.init(*param))) {
          op->dump();
          llvm_unreachable("op inferece init failed");
        }
        inference_map[name] = param;
      }
    });
  }
}

void ModuleInterpreter::allocate_small_tensor_in_mem() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  num_infer_op = 0;
  for (auto func : module.getOps<FuncOp>()) {
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          collect_tensor(v);
          auto name = module::getName(v).str();
          output_names.push_back(name);
        }
      } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto v = in_op.getOutput();
        collect_tensor(v);
        auto name = module::getName(v).str();
        input_names.push_back(name);
      } else if (auto wOp = dyn_cast<top::WeightOp>(op)) {
        auto v = wOp.getOutput();
        auto name = module::getName(v).str();
        mem_map[name] = wOp.read_as_float();
        all_weight_names.push_back(name);
        value_map[name] = v;
      } else if (is_no_mem_op(op)) {
        auto v = op->getResult(0);
        auto name = module::getName(v).str();
        auto in = module::getName(op->getOperand(0)).str();
        mem_map[name] = mem_map[in];
        all_tensor_names.push_back(name);
        value_map[name] = v;
      } else {
        for (auto r : op->getResults()) {
          auto count = module::getNumElements(r);
          auto name = module::getName(r).str();
          if (count >= 6127616) {
            // skip op larger than 187MB
            continue;
          }
          collect_tensor(r);
        }
      }
    });
    module::detachWeightFile(); // to free weight memory

    // input output buffers for all ops
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        if (check_op_in_mem(op)) {
          num_infer_op++;
          auto name = module::getName(op).str();
          auto param = std::make_shared<InferenceParameter>();
          for (auto result : op->getResults()) {
            if (result.getType().isa<NoneType>()) {
              param->outputs.push_back(nullptr);
            } else {
              auto o_name = module::getName(result).str();
              auto data = mem_map[o_name];
              param->outputs.push_back(data->data());
            }
          }
          for (auto input : op->getOperands()) {
            if (module::isNone(input)) {
              param->inputs.push_back(nullptr);
              continue;
            }
            auto input_name = module::getName(input).str();
            if (mem_map.find(input_name) == mem_map.end()) {
              input.dump();
              llvm_unreachable("input operands not allocated");
            } else {
              param->inputs.push_back(mem_map[input_name]->data());
            }
          }
          LLVM_DEBUG(llvm::dbgs() << "init: '" << name << "'\n");
          if (failed(infer_op.init(*param))) {
            op->dump();
            llvm_unreachable("op inferece init failed");
          }
          inference_map[name] = param;
        }
      }
    });
  }
}

void ModuleInterpreter::fake_quant_weight() {
  module::init(module);
  LLVM_DEBUG(llvm::errs() << "start fake_quant_weight\n");
  std::vector<std::string> not_quant_weight_names;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (isa<top::ConvOp>(op) || isa<top::MatMulOp>(op)) {
        auto bias_op = op->getOperands()[2].getDefiningOp();
        if (auto weight_op = dyn_cast<top::WeightOp>(bias_op)) {
          not_quant_weight_names.push_back(module::getName(bias_op).str());
        }
      }
    });
  }

  for (auto &name : all_weight_names) {
    if (std::count(not_quant_weight_names.begin(), not_quant_weight_names.end(),
                   name)) {
      continue;
    }

    auto mem = *mem_map.at(name);
    auto max_value =
        std::max(std::abs(*std::max_element(mem.begin(), mem.end())),
                 std::abs(*std::min_element(mem.begin(), mem.end())));
    for (auto &data : mem) {
      data = std::round(data * 127 / max_value) * max_value / 127;
    }
  }
}

void ModuleInterpreter::invoke(bool express_type) {
  switch (mem_mode) {
  case mem_mode_t::ALL_TENSOR_IN_MEM:
  case mem_mode_t::ALL_TENSOR_IN_REUSED_MEM:
    invoke_all_in_mem(express_type);
    break;
  case mem_mode_t::PART_TENSOR_IN_MEM:
  case mem_mode_t::PART_SMALL_TENSOR_IN_MEM:
    invoke_part_in_mem(express_type);
    break;

  default:
    llvm_unreachable("Mem not enough, please use invoke_to_disk");
    break;
  }
}

void ModuleInterpreter::invoke_all_in_mem(bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  int flag = 0;
  std::string if_name, loop_name;
  for (auto func : module.getOps<FuncOp>()) {
    [[maybe_unused]] WalkResult result = func.walk<
        WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<func::FuncOp>(*op) ||
          isa<top::LoopOp, tpu::LoopOp>(op->getParentOp())) {
        return WalkResult::advance();
      }

      std::string name;
      if (op->getLoc().isa<NameLoc>() || op->getLoc().isa<FusedLoc>()) {
        name = module::getName(op).str();
      }
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << op << "'\n");
      if (flag && isa<func::FuncOp>(*(op->getParentOp()))) {
        flag = 0; // clear
      }
      if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto v = in_op.getOutput();
        auto name = module::getName(v).str();
        call_before_hook(name);
        call_after_hook(name);
        return WalkResult::advance();
      }
      if (isa<tpu::IfOp, top::IfOp>(op)) {
        std::optional<RegisteredOperationName> info =
            op->getName().getRegisteredInfo();
        if_name = name;
        auto *inferInterface =
            info->getInterface<tpu_mlir::InferenceInterface>();
        call_before_hook(name);
        if (failed(inferInterface->inference(inferInterface, op,
                                             *inference_map[name]))) {
          flag = 2; // else branch
        } else {
          flag = 1; // then branch
        }
        call_after_hook(name);
        return WalkResult::advance();
      } else if (isa<top::LoopOp, tpu::LoopOp>(op)) {
        if (isa<top::NoneOp>(op->getOperand(0).getDefiningOp())) {
          if (isa<top::WeightOp>(op->getOperand(1).getDefiningOp()) &&
              cast<top::WeightOp>(op->getOperand(1).getDefiningOp())
                      .read_as_float()
                      ->data()[0] == 1.0f) {
            flag = 3; // do_while
          } else
            flag = 4; // while
        }

        if (isa<top::NoneOp>(op->getOperand(1).getDefiningOp())) {
          flag = 5; // for
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
          flag = 6;
        }

        if (isa<top::NoneOp>(op->getOperand(0).getDefiningOp()) &&
            isa<top::NoneOp>(op->getOperand(0).getDefiningOp())) {
          /* input (\"\", \"\"):
            for (int i=0; ; ++i) {
              cond = ... // Note this value is ignored, but is required in the
            body
            }
          */
          flag = 7; // loop forerver
          llvm_unreachable(
              "fatal error(loop forerver), please modify the origin model");
        }

        loop_name = name;
        Block *bodyBlock;
        // Block &bodyBlock = cast<top::LoopOp>(op).getBody().front();

        llvm::TypeSwitch<Operation *>(op).Case<top::LoopOp, tpu::LoopOp>(
            [&](auto op_) { bodyBlock = &(op_.getBody().front()); });

        auto result_index = [&](Operation *op, int k) -> std::size_t {
          int index = 0;
          const auto &results = op->getOperand(k).getDefiningOp()->getResults();
          if (results.size() >= 2) {
            for (int i = 0; i < results.size(); i++) {
              if (results[i] == op->getOperand(k)) {
                index = i;
              }
            }
          }
          return index;
        };

        using NEW_TYPE =
            std::vector<std::pair<std::size_t, std::unique_ptr<float[]>>>;
        auto execute_body = [&](Block &bodyBlock, std::size_t &cond,
                                const std::string &loop_name, const int l,
                                Operation *const &op,
                                NEW_TYPE &backup) -> void {
          bodyBlock.walk<WalkOrder::PreOrder>([&](Operation *op_) {
            if (auto infer_op = dyn_cast<InferenceInterface>(op_)) {
              std::string op_name;
              if (op_->getLoc().isa<NameLoc>() ||
                  op_->getLoc().isa<FusedLoc>()) {
                op_name = module::getName(op_).str();
              }
              LLVM_DEBUG(llvm::dbgs() << "compute: '" << op_ << "'\n");
              call_before_hook(name);
              if (failed(infer_op.inference(*inference_map[op_name]))) {
                infer_op.dump();
                llvm_unreachable("invoke failed!!");
              }
              call_after_hook(name);
            }

            // move data to LoopOp's result
            if (isa<tpu::YieldOp, top::YieldOp>(op_)) {
              for (int k = 0; k < op_->getNumOperands(); k++) {
                if (!k) {
                  if (!isa<BlockArgument>(op_->getOperand(k))) {
                    auto name =
                        module::getName(op_->getOperand(k).getDefiningOp())
                            .str();
                    cond = inference_map[name]->outputs[0][0];
                  }
                } else {
                  auto num_element = module::getNumElements(op_->getOperand(k));
                  // can return the argument
                  if (isa<BlockArgument>(op_->getOperand(k))) {
                    auto index =
                        op_->getOperand(k).cast<BlockArgument>().getArgNumber();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
                    for (int i = 0; i < num_element; i++) {
                      inference_map[loop_name]->outputs[k - 1][i] =
                          inference_map[loop_name]->inputs[index][i];
                    }
                  } else {
                    // Op maybe have more than 2 results
                    auto index = result_index(op_, k);
                    auto name =
                        module::getName(op_->getOperand(k).getDefiningOp())
                            .str();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
                    for (int i = 0; i < num_element; i++) {
                      inference_map[loop_name]->outputs[k - 1][i] =
                          inference_map[name]->outputs[index][i];
                    }
                  }
                }
              }

              /* update the argument because of
                 loop-carried-dependency for next iteration */
              int Initial_V_size = 0;
              llvm::TypeSwitch<Operation *>(op).Case<top::LoopOp, tpu::LoopOp>(
                  [&](auto Op) { Initial_V_size = Op.getVInitial().size(); });

              for (int k = 0; k < Initial_V_size; k++) {
                if (!l) {
                  // backup the data, for later compare
                  if (!isa<BlockArgument>(op_->getOperand(k + 1)) &&
                      !isa<top::WeightOp>(
                          op->getOperand(k + 2).getDefiningOp())) {
                    auto name =
                        module::getName(op->getOperand(k + 2).getDefiningOp())
                            .str();
                    auto num_element =
                        module::getNumElements(op_->getOperand(k + 1));
                    std::unique_ptr<float[]> data{
                        std::make_unique<float[]>(num_element)};
                    auto index = result_index(op, k + 2);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
                    for (int m = 0; m < num_element; m++) {
                      data[m] = inference_map[name]->outputs[index][m];
                    }
                    backup.emplace_back(k, std::move(data));
                  }
                }

                if (!isa<BlockArgument>(op_->getOperand(k + 1))) {
                  // update the data for next iteration
                  auto src_name =
                      module::getName(op_->getOperand(k + 1).getDefiningOp())
                          .str();
                  auto src_index = result_index(op_, k + 1);
                  auto num_element =
                      module::getNumElements(op_->getOperand(k + 1));

                  Value::user_iterator begin =
                      bodyBlock.getArgument(k + 2).user_begin();
                  Value::user_iterator end =
                      bodyBlock.getArgument(k + 2).user_end();
                  for (; begin != end; begin++) {
                    auto dst_name = module::getName(*begin).str();

                    std::size_t dst_index = 0;
                    for (int p = 0; p < (*begin)->getOperands().size(); p++) {
                      if ((*begin)->getOperand(p) ==
                          bodyBlock.getArgument(k + 2)) {
                        dst_index = p;
                      }
                    }

#pragma omp parallel for schedule(static, omp_schedule(num_element))
                    for (int m = 0; m < num_element; m++) {
                      inference_map[dst_name]->inputs[dst_index][m] =
                          inference_map[src_name]->outputs[src_index][m];
                    }
                  }
                }
              }
            }
            return WalkResult::advance();
          });
        };

        auto restore_data =
            [&](NEW_TYPE &backup,
                std::map<std::string, std::shared_ptr<InferenceParameter>>
                    &inference_map) {
              // restore the data
              for (int kk = 0; kk < backup.size(); kk++) {
                std::size_t operand_index;
                std::unique_ptr<float[]> data;
                std::tie(operand_index, data) = std::move(backup[kk]);
                auto dst_name =
                    module::getName(
                        op->getOperand(operand_index).getDefiningOp())
                        .str();
                auto num_element =
                    module::getNumElements(op->getOperand(operand_index));
                auto dst_index = result_index(op, operand_index);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
                for (int m = 0; m < num_element; m++) {
                  inference_map[dst_name]->outputs[dst_index][m] = data[m];
                }
              }
            };

        auto no_loop_handle = [&](Operation *op, const std::string &loop_name) {
          int number = op->getResults().size() > (op->getNumOperands() - 2)
                           ? (op->getNumOperands() - 2)
                           : op->getResults().size();
          for (int32_t i = 0; i < number; i++) {
            auto num_element = module::getNumElements(op->getOperand(i + 2));
#pragma omp parallel for schedule(static, omp_schedule(num_element))
            for (int k = 0; k < num_element; k++) {
              inference_map[loop_name]->outputs[i][k] =
                  inference_map[loop_name]->inputs[i + 2][k];
            }
          }
        };

        if (flag == 3) {
          std::size_t cond =
              (std::size_t)((*inference_map[loop_name]).inputs[1][0]);
          NEW_TYPE backup;
          int l = 0;
          do {
            execute_body(*bodyBlock, cond, loop_name, l, op, backup);
            l = 1;
          } while (cond);
          restore_data(backup, inference_map);
        } else if (flag == 4) {
          std::size_t cond =
              (std::size_t)((*inference_map[loop_name]).inputs[1][0]);
          NEW_TYPE backup;
          int l = 0;
          int no_loop = cond ? 0 : 1;
          while (cond) {
            execute_body(*bodyBlock, cond, loop_name, l, op, backup);
            l = 1;
          };

          // align with onnxruntime
          if (no_loop) {
            no_loop_handle(op, loop_name);
          }

          restore_data(backup, inference_map);
        } else if (flag == 5) {
          std::size_t trip_count = (*inference_map[loop_name]).inputs[0][0];
          std::size_t cond = 1;
          NEW_TYPE backup;
          int no_loop = trip_count ? 0 : 1;
          for (int l = 0; l < trip_count; l++) {
            execute_body(*bodyBlock, cond, loop_name, l, op, backup);
          }

          // align with onnxruntime
          if (no_loop) {
            no_loop_handle(op, loop_name);
          }

          restore_data(backup, inference_map);
        } else if (flag == 6) {
          std::size_t trip_count = (*inference_map[loop_name]).inputs[0][0];
          std::size_t cond =
              (std::size_t)((*inference_map[loop_name]).inputs[1][0]);
          NEW_TYPE backup;
          int no_loop = (cond && trip_count) ? 0 : 1;
          for (int l = 0; l < trip_count && cond; l++) {
            execute_body(*bodyBlock, cond, loop_name, l, op, backup);
          }

          // align with onnxruntime
          if (no_loop) {
            no_loop_handle(op, loop_name);
          }

          restore_data(backup, inference_map);
        } else {
          llvm_unreachable("other loop mode: Todo");
        }
        // other loop mode: Todo
        return WalkResult::advance();
      } else if (isa<tpu_mlir::InferenceInterface>(op) && 0 == flag) {
        bar.update();
        auto infer_op = dyn_cast<InferenceInterface>(op);
        call_before_hook(name);
        if (failed(infer_op.inference(*inference_map[name]))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
        call_after_hook(name);
      } else if (flag && op->getParentRegion()->getRegionNumber() == flag - 1) {
        if (auto infer_op = dyn_cast<InferenceInterface>(op)) {
          call_before_hook(name);
          if (failed(infer_op.inference(*inference_map[name]))) {
            infer_op.dump();
            llvm_unreachable("invoke failed!!");
          }
          call_after_hook(name);
        }

        if (isa<tpu::YieldOp, top::YieldOp>(op)) {
          for (int k = 0; k < op->getNumOperands(); k++) {
            auto num_element = module::getNumElements(op->getOperand(k));
            name = module::getName(op->getOperand(k).getDefiningOp()).str();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
            for (int i = 0; i < num_element; i++)
              inference_map[if_name]->outputs[k][i] =
                  inference_map[name]->outputs[k][i];
          }
        }
      }

      return WalkResult::advance();
    });
  }
  llvm::errs() << "\n";
  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    for (auto &name : all_tensor_names) {
      auto value = value_map.at(name);
      if (is_no_mem_op(value.getDefiningOp())) {
        continue;
      }
      auto mem = mem_map.at(name);
      if (module::isUniformQuantized(value)) {
        auto qtype = module::getUniformQuantizedType(value);
        for (auto &data : *mem) {
          data = (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
        }
      } else if (module::isCalibratedType(value) &&
                 module::getStorageType(value).isFloat8E4M3FN()) {
        auto qtype = module::getCalibratedType(value);
        for (auto &data : *mem)
          data = (data * qtype.getMax() / get_f8e4m3_max());
      }
    }
  }
}

void ModuleInterpreter::value_to_disk(const std::string &filename,
                                      const std::string &name,
                                      std::vector<float> &data,
                                      bool express_type) {
  // auto value = value_map.at(name);
  // if (express_type && module::isState(module::State::TPU_LOWERED)) {
  //   if (module::isUniformQuantized(value)) {
  //     auto qtype = module::getUniformQuantizedType(value);
  //     for (auto &d : data) {
  //       d = (d - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
  //     }
  //   }
  // }
  // cnpy::npz_save(filename, name, data, "a");
  llvm_unreachable("Not Implemented");
}

void ModuleInterpreter::invoke_to_disk(const std::string &filename,
                                       bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  std::map<std::string, int> mem_uses;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      bar.update();
      tpu_mlir::InferenceParameter p;
      std::vector<std::string> to_free;
      for (auto in : infer_op->getOperands()) {
        if (module::isNone(in)) {
          p.inputs.push_back(nullptr);
          continue;
        }
        auto name = module::getName(in).str();
        if (mem_map.find(name) == mem_map.end()) {
          in.dump();
          llvm_unreachable("input operands not allocated");
        } else {
          p.inputs.push_back(mem_map[name]->data());
        }
        auto iter = mem_uses.find(name);
        if (iter == mem_uses.end()) {
          continue;
        }
        iter->second--;
        if (iter->second == 0) {
          to_free.push_back(name);
        }
      }
      for (auto out : infer_op->getResults()) {
        if (module::isNone(out)) {
          p.outputs.push_back(nullptr);
          continue;
        }
        auto name = module::getName(out).str();
        auto mem_iter = mem_map.find(name);
        if (mem_iter != mem_map.end()) {
          p.outputs.push_back(mem_iter->second->data());
          continue;
        }
        auto count = module::getNumElements(out);
        auto mem = std::make_shared<std::vector<float>>(count);
        mem_map[name] = mem;
        p.outputs.push_back(mem->data());
        int num_uses = std::distance(out.user_begin(), out.user_end());
        mem_uses[name] = num_uses;
        if (num_uses == 0) {
          to_free.push_back(name);
        }
      }
      if (failed(infer_op.init(p))) {
        infer_op.dump();
        llvm_unreachable("init failed!!");
      }
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
      if (failed(infer_op.inference(p))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
      for (auto &m : to_free) {
        value_to_disk(filename, m, *mem_map[m], express_type);
        mem_map.erase(m);
      }
      infer_op.deinit(p);
    });
  }
  llvm::errs() << "\n";
  for (auto &m : all_tensor_names) {
    value_to_disk(filename, m, *mem_map[m], express_type);
  }
}

void ModuleInterpreter::invoke_part_in_mem(bool express_type) {
  module::init(module);
  progressbar bar(num_infer_op);
  std::map<std::string, int> mem_uses;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      bar.update();
      auto name = module::getName(infer_op).str();
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
      if (inference_map.find(name) != inference_map.end()) {
        if (failed(infer_op.inference(*inference_map[name]))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
      } else {
        tpu_mlir::InferenceParameter p;
        std::vector<std::string> to_free;
        for (auto in : infer_op->getOperands()) {
          if (module::isNone(in)) {
            p.inputs.push_back(nullptr);
            continue;
          }
          auto name = module::getName(in).str();
          if (mem_map.find(name) == mem_map.end() ||
              mem_map[name].use_count() == 0) {
            in.dump();
            llvm_unreachable("input operands not allocated");
          } else {
            p.inputs.push_back(mem_map[name]->data());
          }
          auto iter = mem_uses.find(name);
          if (iter == mem_uses.end()) {
            continue;
          }
          iter->second--;
          if (iter->second == 0) {
            to_free.push_back(name);
          }
        }
        for (auto out : infer_op->getResults()) {
          if (module::isNone(out)) {
            p.outputs.push_back(nullptr);
            continue;
          }
          auto name = module::getName(out).str();
          auto mem_iter = mem_map.find(name);
          if (mem_iter != mem_map.end() && mem_map[name].use_count() > 0) {
            p.outputs.push_back(mem_iter->second->data());
            continue;
          }
          auto count = module::getNumElements(out);
          auto mem = std::make_shared<std::vector<float>>(count);
          if (mem.use_count() == 0) {
            llvm_unreachable("allocate failed again!");
          }
          mem_map[name] = mem;
          p.outputs.push_back(mem->data());
          int num_uses = std::distance(out.user_begin(), out.user_end());
          mem_uses[name] = num_uses;
          if (num_uses == 0) {
            to_free.push_back(name);
          }
        }
        if (failed(infer_op.init(p))) {
          infer_op.dump();
          llvm_unreachable("init failed!!");
        }
        LLVM_DEBUG(llvm::dbgs() << "compute: '" << infer_op << "'\n");
        if (failed(infer_op.inference(p))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
        for (auto &m : to_free) {
          mem_map.erase(m);
        }
        infer_op.deinit(p);
      }
    });
  }
  llvm::errs() << "\n";
  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    for (auto &name : all_tensor_names) {
      if (value_map.find(name) == value_map.end() ||
          mem_map.find(name) == mem_map.end() ||
          mem_map[name].use_count() == 0) {
        continue;
      }
      auto value = value_map.at(name);
      auto mem = mem_map.at(name);
      if (module::isUniformQuantized(value)) {
        auto qtype = module::getUniformQuantizedType(value);
        for (auto &data : *mem) {
          data = (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
        }
      } else if (module::isCalibratedType(value) &&
                 module::getStorageType(value).isFloat8E4M3FN()) {
        auto qtype = module::getCalibratedType(value);
        for (auto &data : *mem)
          data = (data * (float)qtype.getMax() / get_f8e4m3_max());
      }
    }
  }
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::invoke_at(const std::string op_name) {
  module::init(module);
  if (value_map.find(op_name) == value_map.end()) {
    llvm::errs() << "Can't find op:" << op_name << "\n";
    llvm_unreachable("invoke_at op_name error");
  }
  auto v = value_map[op_name];
  auto op = v.getDefiningOp();
  if (op == nullptr || false == isa<InferenceInterface>(op)) {
    llvm::errs() << "Op :" << op_name << " can't do inference";
    llvm_unreachable("invoke_at infer error");
  }
  auto infer_op = cast<InferenceInterface>(op);
  LLVM_DEBUG(llvm::dbgs() << "invoke at: '" << infer_op << "'\n");
  call_before_hook(op_name);
  if (failed(infer_op.inference(*inference_map[op_name]))) {
    infer_op.dump();
    llvm_unreachable("infer_op.inference failed!!");
  }
  call_after_hook(op_name);
  return getTensor(op_name);
}

void ModuleInterpreter::invoke_from(const std::string op_name) {
  module::init(module);
  bool start_run = false;
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](InferenceInterface infer_op) {
      auto name = module::getName(infer_op).str();
      if (name == op_name) {
        start_run = true;
      }
      LLVM_DEBUG(llvm::dbgs() << "invoke: '" << infer_op << "'\n");
      if (start_run) {
        call_before_hook(name);
        if (failed(infer_op.inference(*inference_map[name]))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
        call_after_hook(name);
      }
    });
  }
}

// this function is specific for learning weight calibration, returns the
// gradent of weight in conv input is the grd of dst, and returns the gradent of
// weight
void ModuleInterpreter::backward_weight_at(const std::string op_name,
                                           const void *dst_grd,
                                           const int dst_grd_len,
                                           const void *weight_grd,
                                           const int weight_grd_len) {
  module::init(module);
  if (value_map.find(op_name) == value_map.end()) {
    llvm::errs() << "Can't find op:" << op_name << "\n";
    llvm_unreachable("invoke_at op_name error");
  }
  auto v = value_map[op_name];
  auto op = v.getDefiningOp();
  if (op == nullptr || !isa<top::ConvOp>(op)) {
    llvm_unreachable("op.type not support backward_weight!!");
  }

  auto back_param = std::make_shared<InferenceParameter>();
  for (auto result : op->getResults()) {
    if (result.getType().isa<NoneType>()) {
      continue;
    }
    auto type = result.getType().cast<RankedTensorType>();
    auto name = module::getName(result).str();
    size_t count = type.getNumElements();
    if (count != dst_grd_len) {
      llvm_unreachable("output size mis-match");
    }
  }
  back_param->inputs.push_back((float *)dst_grd);
  if (auto convop = dyn_cast<top::ConvOp>(op)) {
    auto opd = convop.getFilter();
    if (opd.getType().isa<NoneType>()) {
      llvm_unreachable("op.filter not exist!!");
    }
    auto type = opd.getType().cast<RankedTensorType>();
    size_t count = type.getNumElements();
    if (count != weight_grd_len) {
      llvm_unreachable("weight grd size mis-match!");
    }
  }
  back_param->outputs.push_back((float *)weight_grd);

  if (op == nullptr || false == isa<InferenceInterface>(op)) {
    llvm::errs() << "Op :" << op_name << " can't do backward";
    llvm_unreachable("backward weight error");
  }
  auto infer_op = cast<InferenceInterface>(op);
  LLVM_DEBUG(llvm::dbgs() << "backward at: '" << op_name << "'\n");
  if (failed(infer_op.backward_weight(*inference_map[op_name], *back_param))) {
    infer_op.dump();
    llvm_unreachable("infer_op.backward failed!!");
  }
  return;
}

void ModuleInterpreter::setTensor(const std::string &name, const void *data,
                                  size_t size, std::vector<int64_t> shape, bool is_integer) {
  module::init(module);
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  bool is_activation =
      std::find(all_tensor_names.begin(), all_tensor_names.end(), name) !=
      all_tensor_names.end();
  auto act = it->second;
  auto tensor_size =
      (mem_mode == mem_mode_t::ALL_TENSOR_IN_REUSED_MEM && is_activation)
          ? activation_offset[name].second
          : act->size();
  auto offset =
      (mem_mode == mem_mode_t::ALL_TENSOR_IN_REUSED_MEM && is_activation)
          ? activation_offset[name].first
          : 0;
  if (tensor_size > size) {
    tensor_size = size;
    it->second->resize(size);
    auto it_value = value_map.find(name);
    module::setShape(it_value->second, shape);

  }
  auto value = value_map.at(name);
  if (is_integer == false && module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    float *p = (float *)data;
    for (uint32_t i = 0; i < tensor_size; i++) {
      float d =
          p[i] * (float)(1 / qtype.getScale()) + (float)qtype.getZeroPoint();
      *(&(act->at(i)) + offset) = qtype.isSigned() ? to_int8(d) : to_uint8(d);
    }
    // std::cout << "is interger" << std::endl;
  } else if (is_integer == false && module::isCalibratedType(value) &&
             module::getStorageType(value).isFloat8E4M3FN()) {
    double scale = module::getCalibratedType(value).getMax() / get_f8e4m3_max();
    F8E4M3((const float *)data, act->data() + offset, tensor_size, 1 / scale,
           true);
  } else if (is_integer == false && module::isCalibratedType(value) &&
             module::getStorageType(value).isFloat8E5M2()) {
    F8E5M2((const float *)data, act->data() + offset, tensor_size, 1., true);

  } else {
    memcpy(act->data() + offset, data, size * sizeof(float));
  }
}

bool ModuleInterpreter::hasTensorMem(const std::string &name) {
  auto it = mem_map.find(name);
  if (it == mem_map.end() || it->second.use_count() == 0) {
    return false;
  }
  return true;
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::getTensor(const std::string &name, bool express_type) {
  auto it = mem_map.find(name);
  if (it == mem_map.end() || mem_map[name].use_count() == 0) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensor failed");
  }
  bool is_activation =
      std::find(all_tensor_names.begin(), all_tensor_names.end(), name) !=
      all_tensor_names.end();
  auto act = it->second;
  auto tensor_size =
      (mem_mode == mem_mode_t::ALL_TENSOR_IN_REUSED_MEM && is_activation)
          ? activation_offset[name].second
          : act->size();
  auto offset =
      (mem_mode == mem_mode_t::ALL_TENSOR_IN_REUSED_MEM && is_activation)
          ? activation_offset[name].first
          : 0;

  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    auto value = value_map.at(name);
    if (module::isUniformQuantized(value)) {
      int i = 0;
      auto mem = mem_map.at(name);
      auto data_fp32 = std::make_shared<std::vector<float>>(tensor_size);
      auto qtype = module::getUniformQuantizedType(value);
      // for (auto &data : *mem) {
      //   data_fp32->data()[i++] =
      //       (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
      // }
      for (i = 0; i < tensor_size; i++) {
        data_fp32->data()[i] =
            (*(mem->data() + offset + i) - (float)qtype.getZeroPoint()) *
            (float)qtype.getScale();
      }
      return std::move(data_fp32);
    } else if (module::isCalibratedType(value) &&
               module::getStorageType(value).isFloat8E4M3FN()) {
      int i = 0;
      auto mem = mem_map.at(name);
      auto data_fp32 = std::make_shared<std::vector<float>>(tensor_size);
      auto qtype = module::getCalibratedType(value);
      double scale = qtype.getMax();
      for (i = 0; i < tensor_size; i++) {
        data_fp32->data()[i] =
            (*(mem->data() + offset + i) * (float)scale / get_f8e4m3_max());
      }
      return std::move(data_fp32);
    }
  }
  // bool is_activation =
  // std::find(all_tensor_names.begin(),all_tensor_names.end(), name) !=
  // all_tensor_names.end(); auto act = it->second; auto tensor_size = (mem_mode
  // == mem_mode_t::ALL_TENSOR_IN_REUSED_MEM && is_activation)?
  // activation_offset[name].second:act->size(); auto offset =(mem_mode ==
  // mem_mode_t::ALL_TENSOR_IN_REUSED_MEM && is_activation)?
  // activation_offset[name].first:0; std::shared_ptr<std::vector<float>>
  // tmp(act->data()+offset,act->data()+offset+tensor_size);
  auto tmp = std::make_shared<std::vector<float>>(
      act->data() + offset, act->data() + offset + tensor_size);
  return std::move(tmp);
}

bool ModuleInterpreter::getTensorQuantInfo(const std::string name,
                                           std::string &dtype, float &scale,
                                           int &zp) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    return false;
  }
  auto value = value_map.at(name);
  auto stype = module::getStorageType(value);
  if (module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    scale = qtype.getScale();
    zp = qtype.getZeroPoint();
    if (stype.isSignlessInteger(8) || stype.isUnsignedInteger(8))
      dtype = std::string("U8");
    else if (stype.isSignedInteger(8))
      dtype = std::string("I8");
    else if (stype.isSignlessInteger(16) || stype.isUnsignedInteger(16))
      dtype = std::string("U16");
    else if (stype.isSignedInteger(16))
      dtype = std::string("I16");
    else if (stype.isSignedInteger(32))
      dtype = std::string("I32");
    else if (stype.isSignlessInteger(32) || stype.isUnsignedInteger(32))
      dtype = std::string("U32");
    else {
      dtype = std::string("I4");
    }
  } else if (stype.isa<FloatType>()) {
    if (stype.isF16()) {
      dtype = std::string("F16");
      scale = 1.0;
      zp = 0;
    } else if (stype.isBF16()) {
      dtype = std::string("BF16");
      scale = 1.0;
      zp = 0;
    } else if (stype.isF32()) {
      dtype = std::string("F32");
      scale = 1.0;
      zp = 0;
    } else if (stype.isFloat8E4M3FN()) {
      dtype = std::string("F8E4");
      scale = 1.0;
      zp = 0;
    } else if (stype.isFloat8E5M2()) {
      dtype = std::string("F8E5");
      scale = 1.0;
      // if (module::isCalibratedType(value) &&
      //     module::getStorageType(value).isFloat8E4M3FN()) {
      //   auto qtype = module::getCalibratedType(value);
      //   scale = qtype.getMax() / get_f8e4m3_max();
      // }
      zp = 0;
    } else {
      dtype = std::string("NA");
      scale = 1.0;
      zp = 0;
    }
  } else if (stype.isa<IntegerType>()) {
    if (stype.isSignedInteger(8))
      dtype = std::string("I8");
    else if (stype.isSignlessInteger(8) || stype.isUnsignedInteger(8))
      dtype = std::string("I8"); // FIXME, seems fail to tell i8 from u8
    else if (stype.isSignlessInteger(16) || stype.isUnsignedInteger(16))
      dtype = std::string("U16");
    else if (stype.isSignedInteger(16))
      dtype = std::string("I16");
    else if (stype.isSignedInteger(32))
      dtype = std::string("I32");
    else if (stype.isSignlessInteger(32) || stype.isUnsignedInteger(32))
      dtype = std::string("U32");
    else {
      dtype = std::string("I4");
    }
    scale = 1.0;
    zp = 0;
  } else {
    dtype = std::string("UK");
    scale = 1.0;
    zp = 0;
  }
  return true;
}

llvm::ArrayRef<int64_t>
ModuleInterpreter::getTensorShape(const std::string &name) {
  auto it = value_map.find(name);
  if (it == value_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensorShape failed");
  }
  return it->second.getType().cast<RankedTensorType>().getShape();
}

void ModuleInterpreter::call_before_hook(std::string layer_name) {
  for (auto hook : before_hooks) {
    hook->run(layer_name);
  }
}
void ModuleInterpreter::call_after_hook(std::string layer_name) {
  for (auto hook : after_hooks) {
    hook->run(layer_name);
  }
}

void ModuleInterpreter::clear_hooks() {
  after_hooks.clear();
  before_hooks.clear();
}

void ModuleInterpreter::set_mem_mode(std::string mem_mode_str) {
  if (mem_mode_str == "reused_mem" || mem_mode_str.empty())
    mem_mode = mem_mode_t::ALL_TENSOR_IN_REUSED_MEM;
  if (mem_mode_str == "force_value_mem")
    mem_mode = mem_mode_t::ALL_TENSOR_IN_MEM;
}
} // namespace tpu_mlir
