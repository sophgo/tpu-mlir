//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/ModuleInterpreter.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <llvm/Support/Debug.h>
#include "progressbar.hpp"
#include "cnpy.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

#define DEBUG_TYPE "interpreter"

static const int64_t MAX_COUNT_LIMIT = 0x100000000ll;
namespace tpu_mlir {
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
  }
}

bool ModuleInterpreter::check_op_in_mem(Operation *op) {
  for (auto r : op->getResults()) {
    if (module::isNone(r)) {
      continue;
    } else {
      auto name = module::getName(r).str();
      if (mem_map.find(name) == mem_map.end()) {
        return false;
      }
    }
  }
  for (auto i : op->getOperands()) {
    if (module::isNone(i)) {
      continue;
    } else {
      auto name = module::getName(i).str();
      if (mem_map.find(name) == mem_map.end()) {
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
          } else {
            auto o_name = module::getName(result).str();
            param->outputs.push_back(mem_map[o_name]->data());
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
    invoke_all_in_mem(express_type);
    break;
  case mem_mode_t::PART_TENSOR_IN_MEM:
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
  std::string if_name;
  for (auto func : module.getOps<FuncOp>()) {
    WalkResult result = func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      bar.update();
      if (isa<func::FuncOp>(*op)) {
        return WalkResult::advance();
      }
      std::string name;
      if (op->getLoc().isa<NameLoc>() || op->getLoc().isa<FusedLoc>())
        name = module::getName(op).str();
      LLVM_DEBUG(llvm::dbgs() << "compute: '" << op << "'\n");
      if (flag && isa<func::FuncOp>(*(op->getParentOp())))
        flag = 0; //clear
      if (isa<tpu::IfOp, top::IfOp>(op)) {
        std::optional<RegisteredOperationName> info = op->getName().getRegisteredInfo();
        if_name = name;
        auto *inferInterface = info->getInterface<tpu_mlir::InferenceInterface>();
        if (failed(inferInterface->inference(inferInterface, op, *inference_map[name]))) {
          flag = 2; //else branch
        } else {
          flag = 1;//then branch
        }
        return WalkResult::advance();
      } else if (isa<tpu_mlir::InferenceInterface>(op) && !flag) {
        auto infer_op = dyn_cast<InferenceInterface>(op);
        if (failed(infer_op.inference(*inference_map[name]))) {
          infer_op.dump();
          llvm_unreachable("invoke failed!!");
        }
      } else if (flag && op->getParentRegion()->getRegionNumber() == flag - 1) {
        if (auto infer_op = dyn_cast<InferenceInterface>(op)) {
          if (failed(infer_op.inference(*inference_map[name]))) {
            infer_op.dump();
            llvm_unreachable("invoke failed!!");
          }
        }

        if (isa<tpu::YieldOp, top::YieldOp>(op)) {
          auto num_element = module::getNumElements(op->getOperand(0));
          name = module::getName(op->getOperand(0).getDefiningOp()).str();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
          for (int i = 0; i < num_element; i++)
            inference_map[if_name]->outputs[0][i] = inference_map[name]->outputs[0][i];
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
          mem_map.erase(m);
        }
        infer_op.deinit(p);
      }
    });
  }
  llvm::errs() << "\n";
  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    for (auto &name : all_tensor_names) {
      auto value = value_map.at(name);
      auto mem = mem_map.at(name);
      if (module::isUniformQuantized(value)) {
        auto qtype = module::getUniformQuantizedType(value);
        for (auto &data : *mem) {
          data = (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
        }
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
  if (failed(infer_op.inference(*inference_map[op_name]))) {
    infer_op.dump();
    llvm_unreachable("infer_op.inference failed!!");
  }

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
      if (start_run && failed(infer_op.inference(*inference_map[name]))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
    });
  }
}
void ModuleInterpreter::setTensor(const std::string &name, const void *data,
                                  size_t size, bool is_integer) {
  module::init(module);
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  auto act = it->second;
  if (act->size() * sizeof(float) != size) {
    llvm::errs() << "Tensor " << name
                 << " data need size: " << act->size() * sizeof(float)
                 << " , but set size: " << size << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  auto value = value_map.at(name);
  if (is_integer == false && module::isUniformQuantized(value)) {
    auto qtype = module::getUniformQuantizedType(value);
    float *p = (float *)data;
    for (uint32_t i = 0; i < act->size(); i++) {
      float d =
          p[i] * (float)(1 / qtype.getScale()) + (float)qtype.getZeroPoint();
      act->at(i) = qtype.isSigned() ? to_int8(d) : to_uint8(d);
    }
  } else {
    memcpy(act->data(), data, size);
  }
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::getTensor(const std::string &name, bool express_type) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, getTensor failed");
  }

  if (express_type && module::isState(module::State::TPU_LOWERED)) {
    auto value = value_map.at(name);
    if (module::isUniformQuantized(value)) {
      int i = 0;
      auto mem = mem_map.at(name);
      auto data_fp32 = std::make_shared<std::vector<float>>(it->second->size());
      auto qtype = module::getUniformQuantizedType(value);
      for (auto &data : *mem) {
        data_fp32->data()[i++] =
            (data - (float)qtype.getZeroPoint()) * (float)qtype.getScale();
      }
      return std::move(data_fp32);
    }
  }

  std::shared_ptr<std::vector<float>> tmp(it->second);
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
  }
  else {
    dtype = std::string("UK"); scale = 1.0; zp = 0;
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

} // namespace tpu_mlir
