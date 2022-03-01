
#include "sophgo/ModuleInterpreter.h"
#include "sophgo/Support/TensorFile.h"
#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>

namespace mlir {

ModuleInterpreter::~ModuleInterpreter() {
  for (auto func : module.getOps<FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<mlir::InferenceInterface>(op)) {
        auto name = op->getAttrOfType<StringAttr>("name").str();
        infer_op.deinit(*inference_map[name]);
      }
    });
  }
}

void ModuleInterpreter::allocate_resources() {
  auto weight_file =
      module->getAttrOfType<StringAttr>("tops.weight_file").str();
  auto wfile = openTensorFile(weight_file);
  for (auto func : module.getOps<FuncOp>()) {
    // if (func.getName() != "main") {
    //   continue;
    // }
    // alloce buffer for all value
    func.walk([&](Operation *op) {
      if (op == func.getOperation()) {
        // self
      } else if (isa<ReturnOp>(op)) {
        for (auto v : op->getOperands()) {
          auto opd = v.getDefiningOp();
          auto name = opd->getAttrOfType<StringAttr>("name").str();
          output_names.push_back(name);
        }
      } else {
        auto result = op->getResult(0);
        auto type = result.getType().cast<TensorType>();
        auto count = type.getNumElements();
        auto name = op->getAttrOfType<StringAttr>("name").str();
        if (isa<tops::WeightOp>(op)) {
          mem_map[name] = wfile->readTensor<float>(name, type);
        } else {
          mem_map[name] = std::make_shared<std::vector<float>>(count);
          value_map[name] = result;
        }
        if (isa<tops::InputOp>(op)) {
          input_names.push_back(name);
        }
      }
    });
    // input output buffers for all ops
    func.walk([&](Operation *op) {
      if (auto infer_op = llvm::dyn_cast<mlir::InferenceInterface>(op)) {
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
    func.walk([&](mlir::InferenceInterface infer_op) {
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

std::vector<std::string> ModuleInterpreter::getAllTensorName() {
  std::vector<std::string> ret;
  for (auto &kv : value_map) {
    ret.push_back(kv.first);
  }
  return ret;
}

} // namespace mlir
