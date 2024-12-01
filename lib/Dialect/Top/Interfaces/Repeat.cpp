//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RepeatOp::getFLOPs() { return 0; }

LogicalResult top::RepeatOp::init(InferenceParameter &p) { return success(); }
void top::RepeatOp::deinit(InferenceParameter &p) {}

LogicalResult top::RepeatOp::inference(InferenceParameter &p) {
  auto repeat_op = getRepeats().getDefiningOp<top::WeightOp>();
  auto repeats = repeat_op.read<float>();
  auto in_shape = module::getShape(getInput());
  auto num_output = module::getNumElements(getOutput());
  int64_t dim = std::max(in_shape.size(), (*repeats).size());
  auto in_shape_ = shape_expand_dim(in_shape, dim);
  auto repeats_ = shape_expand_dim(*repeats, dim);
  int tile_dims = 0;
  for (auto r : repeats_) {
    if (r > 1) {
      tile_dims++;
    }
  }
  std::shared_ptr<std::vector<float>> buffer;
  float *ptr0 = p.outputs[0];
  float *ptr1 = p.outputs[0];
  if (tile_dims > 1) {
    buffer = std::make_shared<std::vector<float>>(num_output);
    ptr1 = buffer->data();
  }
  if (tile_dims % 2 != 0) {
    std::swap(ptr0, ptr1);
  }
  memcpy(ptr0, p.inputs[0], num_output * sizeof(float));
  if (tile_dims == 0) {
    return success();
  }
  std::vector<int64_t> shape(in_shape_);
  for (int i = 0; i < dim; i++) {
    auto r = repeats_[i];
    if (r <= 1) {
      continue;
    }
    function_tile(ptr0, ptr1, shape, i, r);
    shape[i] *= r;
    std::swap(ptr0, ptr1);
  }
  return success();
}

void top::RepeatOp::shape_inference() {
  std::vector<int64_t> repeats;
  if (auto tile_w = dyn_cast<top::WeightOp>(getRepeats().getDefiningOp())) {
    auto tile_v = tile_w.read_as_float();
    std::transform(tile_v->begin(), tile_v->end(), std::back_inserter(repeats),
                   [](auto &v) { return static_cast<int64_t>(v); });
  } else if (module::isShape(getRepeats())) {
    repeats = module::getShapeTensorValue(getRepeats());
  } else {
    llvm_unreachable("repeats is illegal");
  }
  auto in_shape = module::getShape(getInput());
  int64_t dim = std::max(in_shape.size(), repeats.size());
  auto in_shape_ = shape_expand_dim(in_shape, dim);
  auto repeats_ = shape_expand_dim(repeats, dim);
  std::vector<int64_t> out_shape;
  for (int i = 0; i < dim; ++i) {
    out_shape.push_back(in_shape_[i] * repeats_[i]);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    auto input_shape_v = module::getShapeTensorValue(getInput());
    input_shapes_v.push_back(input_shape_v);
    input_shapes_v.push_back(repeats_);
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
