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
  auto input_shape = module::getShape(getInput());
  // calc dims
  int num_axes = axes_val->size();
  std::vector<std::vector<int64_t>> axes_slice;
  std::vector<int64_t> _axes = {axes_val->at(0)};
  for (int i = 1; i < num_axes; i++) {
    if (axes_val->at(i) != axes_val->at(i - 1) + 1) {
      axes_slice.push_back(_axes);
      _axes.clear();
    }
    _axes.push_back(axes_val->at(i));
  }
  axes_slice.push_back(_axes);

  auto tmp_ishape = input_shape.vec();
  int nof_ielmt = std::accumulate(input_shape.begin(), input_shape.end(), 1,
                                  std::multiplies<int64_t>());
  std::vector<float> in(nof_ielmt);
  std::vector<float> out;
  memcpy(in.data(), input_v, in.size() * sizeof(float));
  auto tmp_oshape = tmp_ishape;
  for (int i = axes_slice.size() - 1; i >= 0; i--) {
    auto _axes = axes_slice[i];
    int start_axis = _axes.at(0);
    int end_axis = _axes.back() + 1;
    int outer_dims =
        std::accumulate(tmp_ishape.begin(), tmp_ishape.begin() + start_axis, 1,
                        std::multiplies<int64_t>());
    int axis_dims = std::accumulate(tmp_ishape.begin() + start_axis,
                                    tmp_ishape.begin() + end_axis, 1,
                                    std::multiplies<int64_t>());
    int inner_dims =
        std::accumulate(tmp_ishape.begin() + end_axis, tmp_ishape.end(), 1,
                        std::multiplies<int64_t>());

    for (auto idx : _axes) {
      tmp_oshape[idx] = 1;
    }
    int nof_oelmt = std::accumulate(tmp_oshape.begin(), tmp_oshape.end(), 1,
                                    std::multiplies<int64_t>());
    out.resize(nof_oelmt);
    auto in_data = in.data();
    auto out_data = out.data();

    for (int o = 0; o < outer_dims; o++) {
      for (int i = 0; i < inner_dims; i++) {
        if (type_val == "ReduceMean" || type_val == "ReduceSum") {
          float sum = 0.0f;
          if (inner_dims == 1) {
            sum = std::accumulate(in_data + o * axis_dims,
                                  in_data + (o + 1) * axis_dims, 0.0f);
          } else {
            for (int a = 0; a < axis_dims; a++) {
              sum += in_data[o * axis_dims * inner_dims + a * inner_dims + i];
            }
          }
          if (type_val == "ReduceSum") {
            out_data[o * inner_dims + i] = sum;
          } else {
            sum = sum / axis_dims;
            out_data[o * inner_dims + i] = sum;
          }
        } else if (type_val == "ReduceMax" || type_val == "ReduceMin") {
          float target = in_data[o * axis_dims * inner_dims + i];
          for (int a = 1; a < axis_dims; a++) {
            auto v = in_data[o * axis_dims * inner_dims + a * inner_dims + i];
            if (type_val == "ReduceMax" && v > target) {
              target = v;
            } else if (type_val == "ReduceMin" && v < target) {
              target = v;
            }
          }
          out_data[o * inner_dims + i] = target;
        } else if (type_val == "ReduceL2") {
          float sum = 0.0f;
          for (int a = 0; a < axis_dims; a++) {
            sum += std::pow(
                in_data[o * axis_dims * inner_dims + a * inner_dims + i], 2);
          }
          out_data[o * inner_dims + i] = std::pow(sum, 0.5);
        } else if (type_val == "ReduceL1") {
          float sum = 0.0f;
          for (int a = 0; a < axis_dims; a++) {
            sum +=
                fabs(in_data[o * axis_dims * inner_dims + a * inner_dims + i]);
          }
          out_data[o * inner_dims + i] = sum;
        } else if (type_val == "ReduceProd") {
          float target = in_data[o * axis_dims * inner_dims + i];
          for (int a = 1; a < axis_dims; a++) {
            target *= in_data[o * axis_dims * inner_dims + a * inner_dims + i];
          }
          out_data[o * inner_dims + i] = target;
        } else {
          llvm_unreachable("not support now.");
        }
      }
    }
    if (i == 0) {
      memcpy(output_v, out_data, out.size() * sizeof(float));
    } else {
      in.resize(nof_oelmt);
      in = out;
      tmp_ishape = tmp_oshape;
    }
  }
  auto num_dims = input_shape.size();
  std::vector<int64_t> out_shape;
  for (int i = 0; i < num_dims; i++) {
    if (std::find(axes_val->begin(), axes_val->end(), i) != axes_val->end()) {
      if (getKeepdims()) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(input_shape[i]);
    }
  }
  /* keepdims = false, reduce at all axis,
    it need to set the shape to [1] */
  if (!out_shape.size())
    out_shape.push_back(1);
  module::setShape(getOutput(), out_shape);
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
  if (axes->size() == 0) {
    // for onnx whithout axes attr
    axes->resize(num_dims);
    std::iota(axes->begin(), axes->end(), 0);
    fixed = true;
  }
  std::sort(axes->begin(), axes->end());
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
  if (!out_shape.size()) {
    out_shape.push_back(1);
    auto builder = OpBuilder(getContext());
    setIsScalarAttr(builder.getBoolAttr(true));
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    std::vector<std::vector<int64_t>> input_shapes_v;
    auto input_shape_v = module::getShapeTensorValue(getInput());
    input_shapes_v.push_back(input_shape_v);
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), input_shapes_v, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
