#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <float.h>
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::ReduceOp::getFLOPs() { return Module::getNumElements(output()); }

LogicalResult top::ReduceOp::init(InferenceParameter &p) { return success(); }
void top::ReduceOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReduceOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *output_v = p.outputs[0];
  auto type_val = type().str();
  auto axes_val = Module::getI64Array(axes());
  auto out_shape = Module::getShape(output());
  auto input_shape = Module::getShape(input());
  // calc dims
  int num_dims = input_shape.size();
  int num_axes = axes_val->size();
  int output_dims = out_shape.size();
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
      } else {
        llvm_unreachable("not support now.");
      }
    }
  }
  return success();
}
