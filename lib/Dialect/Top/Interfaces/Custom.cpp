//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cpu_layer.h"
#include "tpu_mlir/Support/CustomLayer.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <dlfcn.h>

int64_t top::CustomOp::getFLOPs() {
  // Flop of CustomOp cannot be determined
  return 0;
}

LogicalResult top::CustomOp::init(InferenceParameter &p) { return success(); }
void top::CustomOp::deinit(InferenceParameter &p) {}

#include "llvm/Support/DynamicLibrary.h"
LogicalResult top::CustomOp::inference(InferenceParameter &p) {
#define MAX_SHAPE_DIMS 8
  const int num_input = getInputs().size();
  std::vector<int[MAX_SHAPE_DIMS]> in_shapes_v(num_input);
  std::vector<int> in_dims_v(num_input);
  for (int i = 0; i < num_input; ++i) {
    auto shape = module::getShape(getInputs()[i]);
    ASSERT_THIS(shape.size() <= MAX_SHAPE_DIMS);
    in_dims_v[i] = shape.size();
    for (int j = 0; j < shape.size(); j++) {
      in_shapes_v[i][j] = shape[j];
    }
  }
  auto params = getParams();
  std::vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "inference_" + op_name;
  llvm::StringRef custom_lib_name = "libplugin_custom.so";
  std::string Err;
  auto custom_dl = llvm::sys::DynamicLibrary::getPermanentLibrary(
      custom_lib_name.data(), &Err);
  ASSERT_THIS(
      custom_dl.isValid() &&
      "Opening libplugin_custom.so failed while using custom op! Check and "
      "make a plugin first");
  if (getName().starts_with("ap.")) {
    // Custom cpu layers
    typedef bmcpu::cpu_layer *(*CreateLayerInstanceFunc)(const char *);
    void *factoryFuncPtr = custom_dl.getAddressOfSymbol("createLayerInstance");
    CreateLayerInstanceFunc createLayerInstance =
        reinterpret_cast<CreateLayerInstanceFunc>(factoryFuncPtr);

    std::transform(
        op_name.begin(), op_name.end(), op_name.begin(),
        [](unsigned char c) -> unsigned char { return std::toupper(c); });
    std::string to_replace = "AP.";
    size_t start_pos = op_name.find(to_replace);
    if (start_pos != std::string::npos) {
      op_name.replace(start_pos, to_replace.size(), "");
    }
    bmcpu::cpu_layer *cpu_layer_instance = createLayerInstance(op_name.c_str());

    std::vector<std::vector<int>> input_shapes_;
    input_shapes_.reserve(num_input);
    vector<vector<int>> output_shapes_;
    for (int i = 0; i < num_input; ++i) {
      auto shape = module::getShape(getInputs()[i]);
      input_shapes_.emplace_back(in_shapes_v[i], in_shapes_v[i] + shape.size());
    }
    cpu_layer_instance->shepe_infer((void *)values.data(),
                                    values.size() * sizeof(custom_param_t),
                                    input_shapes_, output_shapes_);
    cpu_layer_instance->mlir_inference(
        values.data(), values.size() * sizeof(custom_param_t), p.inputs,
        p.outputs, input_shapes_, output_shapes_);
    return success();
  } else {
    auto fPtr = custom_dl.getAddressOfSymbol(api_name.c_str());
    typedef bool (*inference_func_t)(
        void *params, int param_size, int(*input_shapes)[MAX_SHAPE_DIMS],
        int *input_dims, float **inputs, float **outputs);
    auto infer_func = reinterpret_cast<inference_func_t>(fPtr);
    if (infer_func) {
      infer_func(values.data(), values.size() * sizeof(custom_param_t),
                 in_shapes_v.data(), in_dims_v.data(), p.inputs.data(),
                 p.outputs.data());
      return success();
    } else {
      return failure();
    }
  }
}

void top::CustomOp::shape_inference() {
#define MAX_SHAPE_DIMS 8
  const int num_input = getInputs().size();
  const int num_output = getOutputs().size();
  std::vector<int[MAX_SHAPE_DIMS]> in_shapes_v(num_input);
  std::vector<int> in_dims_v(num_input);
  std::vector<int[MAX_SHAPE_DIMS]> out_shapes_v(num_output);
  std::vector<int> out_dims_v(num_output);
  for (int i = 0; i < num_input; ++i) {
    auto shape = module::getShape(getInputs()[i]);
    ASSERT_THIS(shape.size() <= MAX_SHAPE_DIMS);
    in_dims_v[i] = shape.size();
    for (int j = 0; j < shape.size(); j++) {
      in_shapes_v[i][j] = shape[j];
    }
  }
  auto params = getParams();
  std::vector<custom_param_t> values;
  values.push_back({0});
  customOpProcessParam(params, values);
  std::string op_name = getName().str();
  std::string api_name = "shape_inference_" + op_name;
  llvm::StringRef custom_lib_name = "libplugin_custom.so";
  std::string Err;
  auto custom_dl = llvm::sys::DynamicLibrary::getPermanentLibrary(
      custom_lib_name.data(), &Err);
  ASSERT_THIS(
      custom_dl.isValid() &&
      "Opening libplugin_custom.so failed while using custom op! Check and "
      "make a plugin first");
  if (getName().starts_with("ap.")) {
    // Custom cpu layers
    typedef bmcpu::cpu_layer *(*CreateLayerInstanceFunc)(const char *);
    void *factoryFuncPtr = custom_dl.getAddressOfSymbol("createLayerInstance");
    CreateLayerInstanceFunc createLayerInstance =
        reinterpret_cast<CreateLayerInstanceFunc>(factoryFuncPtr);

    std::transform(
        op_name.begin(), op_name.end(), op_name.begin(),
        [](unsigned char c) -> unsigned char { return std::toupper(c); });
    std::string to_replace = "AP.";
    size_t start_pos = op_name.find(to_replace);
    if (start_pos != std::string::npos) {
      op_name.replace(start_pos, to_replace.size(), "");
    }
    bmcpu::cpu_layer *cpu_layer_instance = createLayerInstance(op_name.c_str());

    std::vector<std::vector<int>> input_shapes_;
    input_shapes_.reserve(num_input);
    vector<vector<int>> output_shapes_;
    for (int i = 0; i < num_input; ++i) {
      auto shape = module::getShape(getInputs()[i]);
      input_shapes_.emplace_back(in_shapes_v[i], in_shapes_v[i] + shape.size());
    }

    cpu_layer_instance->shepe_infer((void *)values.data(),
                                    values.size() * sizeof(custom_param_t),
                                    input_shapes_, output_shapes_);
    for (int i = 0; i < num_output; ++i) {
      llvm::SmallVector<int64_t> out_shape(output_shapes_[i].data(),
                                           output_shapes_[i].data() +
                                               output_shapes_[i].size());
      module::setShapeOrVerify(getOutputs()[i], out_shape);
    }
  } else {
    auto fPtr = custom_dl.getAddressOfSymbol(api_name.c_str());
    typedef bool (*shape_infer_func_t)(
        void *params, int param_size, int(*input_shapes)[MAX_SHAPE_DIMS],
        int *input_dims, int(*output_shapes)[MAX_SHAPE_DIMS], int *output_dims);
    auto infer_func = reinterpret_cast<shape_infer_func_t>(fPtr);
    if (infer_func) {
      infer_func(values.data(), values.size() * sizeof(custom_param_t),
                 in_shapes_v.data(), in_dims_v.data(), out_shapes_v.data(),
                 out_dims_v.data());
      for (int i = 0; i < num_output; ++i) {
        llvm::SmallVector<int64_t> out_shape(out_shapes_v[i],
                                             out_shapes_v[i] + out_dims_v[i]);
        module::setShapeOrVerify(getOutputs()[i], out_shape);
      }
    } else {
      llvm::SmallVector<int64_t> out_shape(in_shapes_v[0],
                                           in_shapes_v[0] + in_dims_v[0]);
      module::setShapeOrVerify(getOutputs()[0], out_shape);
    }
  }
}
