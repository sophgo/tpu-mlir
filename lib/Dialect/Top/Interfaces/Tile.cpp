//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::TileOp::getFLOPs() { return 0; }

LogicalResult top::TileOp::init(InferenceParameter &p) { return success(); }
void top::TileOp::deinit(InferenceParameter &p) {}

LogicalResult top::TileOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  auto tmp_size = in_shape.size();
  std::vector<int64_t> tile_vec;
  if (getTile().has_value()) {
    auto tile_v = module::getI64Array(getTile().value());
    tile_vec = *tile_v;
  } else {
    for (int i = 0; i < tmp_size; ++i) {
      tile_vec.emplace_back((int64_t)p.inputs[1][i]);
    }
  }

  int last_i = tile_vec.size() - 1;
  // tile(p.inputs[0], p.outputs[0], in_shape, getAxis(), tile_vec);
  for (int i = 0; i < tile_vec.size(); ++i) {
    last_i = tile_vec.size() - i - 1;
    if (tile_vec[last_i] != 1)
      break;
  }

  auto last_op = p.inputs[0];
  std::vector<int64_t> tmp_shape(in_shape.begin(), in_shape.end());

  for (int i = 0; i < last_i + 1; ++i) {
    if (tile_vec[i] == 1)
      continue;
    int len = std::accumulate(tmp_shape.begin(), tmp_shape.end(), 1,
                              std::multiplies<int64_t>());
    float *cur_input = new float[len];
    std::copy(last_op, last_op + len, cur_input);
    function_tile(cur_input, p.outputs[0],
                  llvm::ArrayRef<int64_t>(tmp_shape.data(), tmp_size), i,
                  tile_vec[i]);
    last_op = p.outputs[0];
    tmp_shape[i] *= tile_vec[i];
    delete[] cur_input;
  }
  // std::cout<<"mlir inference TileOp"<<std::endl;
  return success();
}

void top::TileOp::shape_inference() {
  auto in0_shape = module::getShape(getInput());
  std::vector<int64_t> tile_vec;
  if (getTile().has_value()) {
    auto tile_v = module::getI64Array(getTile().value());
    tile_vec = *tile_v;
  } else if (auto tile_w =
                 dyn_cast<top::WeightOp>(getTileT().getDefiningOp())) {
    auto tile_v = tile_w.read_as_float();
    std::transform(tile_v->begin(), tile_v->end(), std::back_inserter(tile_vec),
                   [](auto &v) { return static_cast<int64_t>(v); });
  } else if (module::isShape(getTileT())) {
    tile_vec = module::getShapeTensorValue(getTileT());
  } else {
    llvm_unreachable("tile_vec is illegal");
  }
  ASSERT_THIS(in0_shape.size() == tile_vec.size());
  std::vector<int64_t> out_shape(in0_shape.size());
  std::transform(tile_vec.begin(), tile_vec.end(), in0_shape.begin(),
                 out_shape.begin(), [](int a, int b) { return a * b; });
  module::setShapeOrVerify(getOutput(), out_shape);

  if (module::isShape(getInput())) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    auto input_shape_v = module::getShapeTensorValue(getInput());
    input_shapes_v.push_back(input_shape_v);
    input_shapes_v.push_back(tile_vec);
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
