#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ReduceOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::ReduceOp::init(InferenceParameter &p) { return success(); }
void top::ReduceOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReduceOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_v = p.outputs[0];
  auto type_val = getMode().str();
  auto axes_val = module::getI64Array(getAxes());
  auto out_shape = module::getShape(getOutput());
  auto input_shape = module::getShape(getInput());
  // calc dims
  int num_dims = input_shape.size();
  int num_axes = axes_val->size();
  for (int i = 1; i < num_axes; i++) {
    assert(axes_val->at(i) == axes_val->at(i - 1) + 1);
    assert(axes_val->at(i) < num_dims);
  }
  int start_axis = axes_val->at(0);
  int end_axis = axes_val->at(num_axes - 1) + 1;
  int outer_dims =
      std::accumulate(input_shape.begin(), input_shape.begin() + start_axis, 1,
                      std::multiplies<int64_t>());
  int axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                  input_shape.begin() + end_axis, 1,
                                  std::multiplies<int64_t>());
  int inner_dims =
      std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                      std::multiplies<int64_t>());
  for (int o = 0; o < outer_dims; o++) {
    for (int i = 0; i < inner_dims; i++) {
      if (type_val == "ReduceMean" || type_val == "ReduceSum") {
        float sum = 0.0f;
        if (inner_dims == 1) {
          sum = std::accumulate(input_v + o * axis_dims,
                                input_v + (o + 1) * axis_dims, 0.0f);
        } else {
          for (int a = 0; a < axis_dims; a++) {
            sum += input_v[o * axis_dims * inner_dims + a * inner_dims + i];
          }
        }
        if (type_val == "ReduceSum") {
          output_v[o * inner_dims + i] = sum;
        } else {
          sum = sum / axis_dims;
          output_v[o * inner_dims + i] = sum;
        }
      } else if (type_val == "ReduceMax" || type_val == "ReduceMin") {
        float target = input_v[o * axis_dims * inner_dims + i];
        for (int a = 1; a < axis_dims; a++) {
          auto v = input_v[o * axis_dims * inner_dims + a * inner_dims + i];
          if (type_val == "ReduceMax" && v > target) {
            target = v;
          } else if (type_val == "ReduceMin" && v < target) {
            target = v;
          }
        }
        output_v[o * inner_dims + i] = target;
      } else if (type_val == "ReduceL2") {
        float sum = 0.0f;
        for (int a = 0; a < axis_dims; a++) {
          sum += std::pow(
              input_v[o * axis_dims * inner_dims + a * inner_dims + i], 2);
        }
        output_v[o * inner_dims + i] = std::pow(sum, 0.5);
      } else if (type_val == "ReduceL1") {
        float sum = 0.0f;
        for (int a = 0; a < axis_dims; a++) {
          sum += fabs(input_v[o * axis_dims * inner_dims + a * inner_dims + i]);
        }
        output_v[o * inner_dims + i] = sum;
      } else if (type_val == "ReduceProd") {
        float target = input_v[o * axis_dims * inner_dims + i];
        for (int a = 1; a < axis_dims; a++) {
          target *= input_v[o * axis_dims * inner_dims + a * inner_dims + i];
        }
        output_v[o * inner_dims + i] = target;
      } else {
        llvm_unreachable("not support now.");
      }
    }
  }
  return success();
}

void top::ReduceOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num_dims = in_shape.size();
  auto axes = module::getI64Array(getAxes());
  std::vector<int64_t> out_shape;
  bool fixed = false;
  for (auto &idx : *axes) {
    if (idx < 0) {
      idx += num_dims;
      fixed = true;
    }
  }
  if (fixed) {
    Builder builder(getContext());
    setAxesAttr(builder.getI64ArrayAttr(*axes));
  }
  for (int i = 0; i < num_dims; i++) {
    if (std::find(axes->begin(), axes->end(), i) != axes->end()) {
      if (getKeepdims()) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  /* keepdims = false, reduce at all axis,
    it need to set the shape to [1] */
  if (!out_shape.size())
    out_shape.push_back(1);
  module::setShapeOrVerify(getOutput(), out_shape);
}
