
#include "sophgo/ModuleInterpreter.h"
#include "sophgo/Support/MathUtils.h"
#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

using namespace mlir;

namespace sophgo {

ModuleInterpreter::~ModuleInterpreter() {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = op->getAttrOfType<StringAttr>("name").str();
        infer_op.deinit(*inference_map[name]);
      }
    });
  }
}

void ModuleInterpreter::allocate_resources() {
  all_tensor_names.clear();
  value_map.clear();
  mem_map.clear();
  for (auto func : module.getOps<FuncOp>()) {
    // if (func.getName() != "main") {
    //   continue;
    // }
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation() || isa<top::NoneOp>(op)) {
        // self
      } else if (isa<func::ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto opd = v.getDefiningOp();
          auto name = opd->getAttrOfType<StringAttr>("name").str();
          output_names.push_back(name);
        }
      } else {
        auto result = op->getResult(0);
        auto type = result.getType().cast<RankedTensorType>();
        auto count = type.getNumElements();
        auto name = op->getAttrOfType<StringAttr>("name").str();
        value_map[name] = result;
        if (auto wOp = llvm::dyn_cast<top::WeightOp>(op)) {
          mem_map[name] =wOp.read<float>();
        } else {
          mem_map[name] = std::make_shared<std::vector<float>>(count);
          all_tensor_names.push_back(name);
        }
        if (isa<top::InputOp>(op)) {
          input_names.push_back(name);
        }
      }
    });
    // input output buffers for all ops
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<InferenceInterface>(op)) {
        auto name = op->getAttrOfType<StringAttr>("name").str();
        auto param = std::make_shared<InferenceParameter>();
        param->outputs.push_back(mem_map[name]->data());
        for (auto input : op->getOperands()) {
          if (input.getType().isa<mlir::NoneType>()) {
            param->inputs.push_back(nullptr);
            continue;
          }
          auto input_name =
              input.getDefiningOp()->getAttrOfType<StringAttr>("name").str();
          if (mem_map.find(input_name) == mem_map.end()) {
            input.dump();
            llvm_unreachable("input operands not allocated");
          } else {
            param->inputs.push_back(mem_map[input_name]->data());
          }
        }
        if (failed(infer_op.init(*param))) {
          op->dump();
          llvm_unreachable("op inferece init failed");
        }
        inference_map[name] = param;
      }
    });
  }
}

void ModuleInterpreter::invoke() {
  for (auto func : module.getOps<mlir::FuncOp>()) {
    // if (func.getName() != "main") {
    //   continue;
    // }
    func.walk([&](InferenceInterface infer_op) {
      auto name = infer_op->getAttrOfType<StringAttr>("name").str();
      if (failed(infer_op.inference(*inference_map[name]))) {
        infer_op.dump();
        llvm_unreachable("invoke failed!!");
      }
    });
  }
}

void ModuleInterpreter::setTensor(const std::string &name, const void *data,
                                  size_t size) {
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
  memcpy(act->data(), data, size);
}

std::shared_ptr<std::vector<float>>
ModuleInterpreter::getTensor(const std::string &name) {
  auto it = mem_map.find(name);
  if (it == mem_map.end()) {
    llvm::errs() << "Can't find op name: " << name << "\n";
    llvm_unreachable("Error, setTensor failed");
  }
  return it->second;
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

} // namespace mlir
