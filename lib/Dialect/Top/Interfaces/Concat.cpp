//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Concat.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ConcatOp::getFLOPs() { return 0; }

LogicalResult top::ConcatOp::init(InferenceParameter &p) {

  auto concat = new Concat();
  auto axis_ = getAxis();
  concat_attr_t attr;
  attr.num_src = getInputs().size();

  for (int i = 0; i < attr.num_src; i++) {
    auto input_shape = module::getShape(getInputs()[i]);
    int channel = input_shape[axis_];

    int outer_dim = 1;
    for (int i = 0; i < axis_; i++) {
      outer_dim *= input_shape[i];
    }
    int inner_dim = 1;
    for (int i = axis_ + 1; i < input_shape.size(); i++) {
      inner_dim *= input_shape[i];
    }

    attr.src_shapes.push_back({outer_dim, channel, inner_dim});
  }

  attr.dst_shape = module::getShape(getOutput());
  attr.axis = 1;
  concat->setup(p.inputs, p.outputs[0], attr);
  p.handle = (void *)concat;

  return success();
}

void top::ConcatOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto concat = (Concat *)(p.handle);
    delete concat;
    p.handle = nullptr;
  }
}

LogicalResult top::ConcatOp::inference(InferenceParameter &p) {

  if (p.handle == nullptr) {
    return failure();
  }
  auto concat = (Concat *)p.handle;
  auto axis_ = getAxis();
  auto in0_shape = module::getShape(getInputs()[0]);
  if (axis_ < 0) {
    axis_ += in0_shape.size();
    setAxis(axis_);
  }
  int64_t shape_axis = 0;
  for (auto inp : getInputs()) {
    auto shape = module::getShape(inp);
    shape_axis += shape[axis_];
  }
  std::vector<int64_t> out_shape(in0_shape);
  out_shape[axis_] = shape_axis;
  module::setShape(getOutput(), out_shape);

  concat_attr_t attr;
  attr.num_src = getInputs().size();
  for (int i = 0; i < attr.num_src; i++) {
    auto input_shape = module::getShape(getInputs()[i]);
    int channel = input_shape[axis_];

    int outer_dim = 1;
    for (int i = 0; i < axis_; i++) {
      outer_dim *= input_shape[i];
    }
    int inner_dim = 1;
    for (int i = axis_ + 1; i < input_shape.size(); i++) {
      inner_dim *= input_shape[i];
    }

    attr.src_shapes.push_back({outer_dim, channel, inner_dim});
  }

  attr.dst_shape = module::getShape(getOutput());
  attr.axis = 1;
  concat->setup(p.inputs, p.outputs[0], attr);
  concat->run();

  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0],
                  module::getNumElements(getOutput()), limit);
  }

  return success();
}

void top::ConcatOp::shape_inference() {
  auto axis_ = getAxis();
  auto in0_shape = module::getShape(getInputs()[0]);
  if (axis_ < 0) {
    axis_ += in0_shape.size();
    setAxis(axis_);
  }
  int64_t shape_axis = 0;
  for (auto inp : getInputs()) {
    auto shape = module::getShape(inp);
    shape_axis += shape[axis_];
  }
  std::vector<int64_t> out_shape(in0_shape);
  out_shape[axis_] = shape_axis;
  module::setShapeOrVerify(getOutput(), out_shape);
  if (llvm::find_if(getOperands(), module::isShape) != getOperands().end()) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    for (const auto &input : getOperands()) {
      if (module::isShape(input)) {
        auto input_shape_v = module::getShapeTensorValue(input);
        input_shapes_v.push_back(input_shape_v);
      } else if (module::isWeight(input)) {
        auto data = input.getDefiningOp<top::WeightOp>().read_as_float();
        std::vector<int64_t> data_v(data->begin(), data->end());
        input_shapes_v.push_back(data_v);
      } else if (module::getShape(input).size() == 1 &&
                 module::getShape(input)[0] == 0) {
        continue;
      } else {
        auto input_shape_v = module::getShape(input);
        input_shapes_v.push_back(input_shape_v);
      }
    }
    ASSERT_THIS(out_shape.size() == 1 || out_shape.size() == 0);
    auto real_out_size = out_shape.size() == 0 ? 1 : out_shape[0];
    InferenceParameter p;
    std::vector<std::vector<float_t>> input_datas;
    for (auto &in_shape_v : input_shapes_v) {
      std::vector<float_t> input_data(in_shape_v.size());
      std::transform(in_shape_v.begin(), in_shape_v.end(), input_data.begin(),
                     [](auto &i) { return static_cast<float_t>(i); });
      input_datas.push_back(input_data);
    }
    std::transform(input_datas.begin(), input_datas.end(),
                   std::back_inserter(p.inputs),
                   [](auto &i) { return i.data(); });
    std::vector<float_t> output_data(real_out_size);
    p.outputs.push_back(output_data.data());
    auto inf_op = dyn_cast<InferenceInterface>(getOperation());
    inf_op.init(p);
    ASSERT_THIS(inf_op);
    auto ret = inf_op.inference(p);
    ASSERT_THIS(mlir::succeeded(ret));
    inf_op.deinit(p);
    std::vector<int64_t> output_shape_v(real_out_size);
    std::transform(output_data.begin(), output_data.end(),
                   output_shape_v.begin(),
                   [](float_t i) { return static_cast<int64_t>(i); });
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
