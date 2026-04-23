#include "tpu_mlir/Support/Dnnl/Dnnl.h"

mlp_attr_t top::MlpOp::parseParam() {
  mlp_attr_t p = {0};
  return p;
}

int64_t top::MlpOp::getFLOPs() { return 0; }

LogicalResult top::MlpOp::init(InferenceParameter &p) { return success(); }

void top::MlpOp::deinit(InferenceParameter &p) { return; }

LogicalResult top::MlpOp::inference(InferenceParameter &p) { return success(); }

void top::MlpOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  ASSERT_THIS(input_shape.size() == 3);
  if (getIsExpert()) {
    auto num_expert_per_tok = getNumExpertPerTok();
    std::vector<int64_t> out_shape = {input_shape[0] * input_shape[1],
                                      static_cast<int64_t>(num_expert_per_tok),
                                      input_shape[2]};
    module::setShapeOrVerify(getOutput(), out_shape);
  } else {
    module::setShapeOrVerify(getOutput(), input_shape);
  }
}
