#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::MaxPoolingIndicesBwdOp::getFLOPs() {
  auto kernel = module::getI64Array(getKernelShape());
  auto dilation = module::getI64Array(getDilations());
  int kh = kernel->at(0);
  int kw = kernel->at(1);
  int kd = dilation->at(0);
  return module::getNumElements(getGradInput()) * (kd * kh * kw);
}

LogicalResult top::MaxPoolingIndicesBwdOp::init(InferenceParameter &p) {
  return success();
}

void top::MaxPoolingIndicesBwdOp::deinit(InferenceParameter &p) { return; }

LogicalResult top::MaxPoolingIndicesBwdOp::inference(InferenceParameter &p) {
  const float *grad_input_data = p.inputs[0];
  const float *indices_data = p.inputs[1];
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pads = module::getI64Array(getPads());
  auto input_shape = module::getI64Array(getInputShape());
  float *output_data = p.outputs[0];
  int n = input_shape->at(0);
  int c = input_shape->at(1);
  int h = input_shape->at(2);
  int w = input_shape->at(3);
  memset(output_data, 0, sizeof(float) * n * c * h * w);
  int kernel_h = kernel->at(0);
  int kernel_w = kernel->at(1);
  int padding_h = pads->at(0);
  int padding_w = pads->at(1);
  int stride_h = stride->at(0);
  int stride_w = stride->at(1);
  auto cur_shape = module::getShape(getGradOutput());
  int cur_h = cur_shape[2];
  int cur_w = cur_shape[3];
  ASSERT_THIS(cur_h == (h + 2 * padding_h - kernel_h) / stride_h + 1);
  ASSERT_THIS(cur_w == (w + 2 * padding_w - kernel_w) / stride_w + 1);
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < cur_h; hi++) {
        for (int wi = 0; wi < cur_w; wi++) {
          int index =
              ni * c * cur_h * cur_w + ci * cur_h * cur_w + hi * cur_w + wi;
          int out_index = ni * c * h * w + ci * h * w;
          int max_index = (int)indices_data[index];
          output_data[out_index + max_index] += grad_input_data[index];
        }
      }
    }
  }
  return success();
}

void top::MaxPoolingIndicesBwdOp::shape_inference() {
  auto input_shape = module::getI64Array(getInputShape());
  std::vector<int64_t> output_shape = {input_shape->at(0), input_shape->at(1),
                                       input_shape->at(2), input_shape->at(3)};
  module::setShapeOrVerify(getGradInput(), output_shape);
}