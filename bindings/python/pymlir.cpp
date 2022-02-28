#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

// -------------
// pure C++ code
// -------------
#include "sophgo/Dialect/Tops/IR/TopsOps.h"
#include "sophgo/ModuleInterpreter.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

#define OP_NAME "name"
#define OP_TYPE "type"
#define OP_QUANT "quant"

typedef std::map<std::string, std::shared_ptr<std::vector<float>>> tensor_map_t;
typedef std::map<std::string, std::vector<int64_t>> shape_map_t;

static bool isValidOp(Operation &op) {
  return (op.getName().getDialect()->getNamespace() == "tpu" &&
          !isa<tpu::WeightFileOp>(op) && !isa<tpu::LoadWeightOp>(op) &&
          !isa<tpu::NoneOp>(op));
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

template <typename Dtype>
static py::array getPythonArray(std::vector<Dtype> *vec,
                                const std::vector<int64_t> &shape) {
  std::vector<unsigned> stride_v(shape.size(), sizeof(Dtype));
  for (int i = shape.size() - 1; i > 0; i--) {
    for (int j = 0; j < i; j++) {
      stride_v[j] *= shape[i];
    }
  }

  return py::array(
      py::buffer_info(vec->data(),   /* data as contiguous array  */
                      sizeof(Dtype), /* size of one scalar        */
                      py::format_descriptor<Dtype>::format(), /* data type */
                      shape.size(),                           // ndim/
                      shape,                                  // shape
                      stride_v                                // strides
                      ));
}

static py::dict getTensorDict(tensor_map_t &tensorMap, shape_map_t &shapeMap) {
  py::dict py_ret;
  for (auto it = tensorMap.begin(); it != tensorMap.end(); it++) {
    auto op = it->first;
    auto data = it->second.get();
    py::str py_s(op);

    assert(shapeMap.end() != shapeMap.find(op));
    py_ret[py_s] = getPythonArray(it->second.get(), shapeMap[op]);
  }

  return py_ret;
}

class py_module {
public:
  py_module() {}
  ~py_module() {
    interpreter_.reset();
    auto module = module_.release();
    if (module) {
      module.erase();
    }
    context.reset();
  }

  void load(std::string filename) {
    if (context) {
      context.reset();
    }

    DialectRegistry registry;
    registry.insert<tops::TopsDialect, StandardOpsDialect>();
    context = std::make_unique<MLIRContext>(registry);

    module_ = parseMLIRInput(filename);
    if (!module_) {
      llvm_unreachable("could not parse the input IR\n");
    }
    if (interpreter_) {
      interpreter_.reset();
    }

    interpreter_ = std::make_unique<ModuleInterpreter>();
    interpreter_->updateWeightMap(module_);
    interpreter_->loadModule(module_);
  }

  OwningModuleRef parseMLIRInput(StringRef inputFilename) {
    // Set up the input file.
    std::string errorMessage;
    auto file = openInputFile(inputFilename, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      llvm_unreachable("read find failed");
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningModuleRef(parseSourceFile(sourceMgr, context.get()));
  }

  void dump(std::string name) { interpreter_->dump(name); }

  py::dict getAllTensor() {
    tensor_map_t tensorMap_;
    shape_map_t shapeMap_;
    auto all_tensor_names = interpreter_->getAllTensorName();
    for (auto &tensor_name : all_tensor_names) {
      tensorMap_[tensor_name] = interpreter_->getTensor(tensor_name, 0);
      shapeMap_[tensor_name] = interpreter_->getTensorShape(tensor_name);
    }

    return getTensorDict(tensorMap_, shapeMap_);
  }
  py::list get_input_names() {
    py::list ret;
    auto &inputs = interpreter_->input_names;
    for (auto &i : inputs) {
      ret.append(i);
    }
    return ret;
  }
  py::list get_output_names() {
    py::list ret;
    auto &outputs = interpreter_->output_names;
    for (auto &i : outputs) {
      ret.append(i);
    }
    return ret;
  }

  void set_tensor(
      std::string name,
      py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    interpreter_->setTensor(name, data.data(), data.size() * sizeof(float));
  }
  py::array get_tensor(std::string name) {
    auto tensor = interpreter_->getTensor(name);
    std::vector<int64_t> shape = interpreter_->getTensorShape(name);
    return getPythonArray(tensor.get(), shape);
  }
  void invoke() { interpreter_->invoke(); }

  // wrap C++ function with NumPy array IO
  py::dict
  run(py::array_t<float, py::array::c_style | py::array::forcecast> array,
      std::string target_op) {
    if (!interpreter_) {
      throw std::runtime_error("Not load mlir Model");
    }
    std::vector<float> input_vec(array.size());
    std::memcpy(input_vec.data(), array.data(), array.size() * sizeof(float));
    std::vector<int64_t> input_shape;
    for (ssize_t i = 0; i < array.ndim(); ++i) {
      input_shape.push_back((int64_t)array.shape()[i]);
    }

    size_t input_size = std::accumulate(input_shape.begin(), input_shape.end(),
                                        1, std::multiplies<int64_t>());
    auto &input_details = interpreter_->input_details;
    size_t all_need_data_size = 0;
    for (auto &i : input_details) {
      all_need_data_size += i.second;
    }
    if (input_size != all_need_data_size) {
      llvm::errs() << "input data size: " << input_size << "\n";
      for (auto &i : input_details) {
        llvm::errs() << i.first << " needed data size: " << i.second << "\n";
      }
      llvm::errs() << "all input needed size: " << all_need_data_size << "\n";
      llvm_unreachable("input data size not same with all input needed size");
    }

    // set tensor
    size_t slice_idx = 0;
    for (auto &i : input_details) {
      std::vector<float> input_data(input_vec.begin() + slice_idx,
                                    input_vec.begin() + slice_idx + i.second);
      slice_idx += i.second;
      interpreter_->setTensor(i.first, input_data, 0);
    }
    assert(slice_idx == input_vec.size());

    tensor_map_t results;
    shape_map_t shapeMap_;
    if (!target_op.empty()) {
      interpreter_->invokeTo(target_op, 0);
      results[target_op] = interpreter_->getTensor(target_op, 0);
      shapeMap_[target_op] = interpreter_->getTensorShape(target_op);
    } else {
      interpreter_->invoke(0);
      auto output_details = interpreter_->output_details;
      for (auto &output_name : output_details) {
        results[output_name] = interpreter_->getTensor(output_name, 0);
        shapeMap_[output_name] = interpreter_->getTensorShape(output_name);
      }
    }
    return getTensorDict(results, shapeMap_);
  }

public:
  py::list opInfo_;
  static std::string version;

private:
  std::unique_ptr<mlir::MLIRContext> context;
  OwningModuleRef module_;
  std::string weightFilePath_;
  std::unique_ptr<ModuleInterpreter> interpreter_;
};

std::string py_module::version = MLIR_VERSION;

// wrap as Python module
PYBIND11_MODULE(pymlir, m) {
  m.doc() = "pybind11 for mlir";

  py::class_<py_module>(m, "module", "MLIR Module")
      .def(py::init<>())
      .def("load", &py_module::load, "load module from IR")
      .def("get_all_tensor", &py_module::getAllTensor, "dump all tensor data")
      .def("set_tensor", &py_module::set_tensor)
      .def("get_tensor", &py_module::get_tensor, "get one tensor data")
      .def_readwrite("op_info", &py_module::opInfo_)
      .def("run", &py_module::run, py::arg("array"), py::arg("target_op") = "",
           "run module inference with input array, and return output array")
      .def("dump", &py_module::dump)
      .def("invoke", &py_module::invoke)
      .def("get_input_details", &py_module::get_input_details)
      .def("get_output_details", &py_module::get_output_details)
      .def_readonly_static("version", &py_module::version);
}
