//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MatchTemplateOp::getFLOPs() {
  int64_t flops = 0;
  auto mode = getMode().str();
  auto input_shape = module::getShape(getInput());
  auto template_shape = module::getShape(getMatch());

  ASSERT_THIS(input_shape.size() == 2);
  ASSERT_THIS(template_shape.size() == 2);
  ASSERT_THIS(input_shape[0] >= template_shape[0]);
  ASSERT_THIS(input_shape[1] >= template_shape[1]);
  auto h = template_shape[0];
  auto w = template_shape[1];
  auto n = input_shape[0] - h + 1;
  auto c = input_shape[1] - w + 1;

  auto outer_size = n * c;
  auto match_size = h * w;
  if (mode == "TM_CCOEFF_NORMED") {
    flops = outer_size * 6 * match_size;
  } else if (mode == "TM_SQDIFF") {
    flops = outer_size * (3 * match_size - 1);
  } else {
    llvm_unreachable("Not support now.");
  }
  return flops;
}

LogicalResult top::MatchTemplateOp::init(InferenceParameter &p) {
  return success();
}

void top::MatchTemplateOp::deinit(InferenceParameter &p) {}

LogicalResult top::MatchTemplateOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *template_v = p.inputs[1];
  float *output_v = p.outputs[0];
  auto mode = getMode().str();
  auto input_shape = module::getShape(getInput());
  auto template_shape = module::getShape(getMatch());

  ASSERT_THIS(mode == "TM_CCOEFF_NORMED" || mode == "TM_SQDIFF");
  ASSERT_THIS(input_shape.size() == 2);
  ASSERT_THIS(template_shape.size() == 2);
  ASSERT_THIS(input_shape[0] >= template_shape[0]);
  ASSERT_THIS(input_shape[1] >= template_shape[1]);

  auto n = input_shape[0] - template_shape[0] + 1;
  auto c = input_shape[1] - template_shape[1] + 1;
  auto h = template_shape[0];
  auto w = template_shape[1];
  auto stride = input_shape[1];
  auto outer_size = n * c;
  auto match_size = h * w;
  // auto scale = 1. / (h * w * 100);
  float_t tmean = 0.;
  if (mode == "TM_CCOEFF_NORMED") {
    for (int i = 0; i < match_size; i++) {
      tmean += template_v[i];
    }
    tmean /= match_size;
  }

#pragma omp parallel for schedule(static, omp_schedule(outer_size))
  for (int i = 0; i < outer_size; i++) {
    uint32_t ioffset = i / c * stride + i % c;
    auto input = input_v + ioffset;
    if (mode == "TM_CCOEFF_NORMED") {
      double_t dividend = 0;
      double_t wndSum2 = 0;
      double_t templSum2 = 0;
      double_t imean = 0;
      for (int32_t i = 0; i < match_size; i++) {
        uint32_t offset = i / w * stride + i % w;
        imean += input[offset];
      }
      imean /= match_size;
      for (int32_t i = 0; i < match_size; i++) {
        uint32_t offset = i / w * stride + i % w;
        auto inp = input[offset] - imean;
        auto tpl = template_v[i] - tmean;
        dividend += inp * tpl;
        wndSum2 += std::pow(inp, 2);
        templSum2 += std::pow(tpl, 2);
      }
      output_v[i] = dividend * std::pow(wndSum2 * templSum2, -0.5);
    } else if (mode == "TM_SQDIFF") {
      double sum = 0;
      for (int32_t i = 0; i < match_size; i++) {
        uint32_t offset = i / w * stride + i % w;
        sum += std::pow(input[offset] - template_v[i], 2);
      }
      // output_v[i] = std::pow(scale * sum + 1e-5, -0.5);
      output_v[i] = sum;
    } else {
      llvm_unreachable("Not support now.");
    }
  }
  return success();
}

void top::MatchTemplateOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto template_shape = module::getShape(getMatch());

  ASSERT_THIS(input_shape.size() == 2);
  ASSERT_THIS(template_shape.size() == 2);
  ASSERT_THIS(input_shape[0] >= template_shape[0]);
  ASSERT_THIS(input_shape[1] >= template_shape[1]);

  auto n = input_shape[0] - template_shape[0] + 1;
  auto c = input_shape[1] - template_shape[1] + 1;
  module::setShapeOrVerify(getOutput(), {n, c});
}
