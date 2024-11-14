//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/LutFunc.h"

LogicalResult tpu::MatchTemplateOp::init(InferenceParameter &p) {
  return success();
}
void tpu::MatchTemplateOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MatchTemplateOp::inference(InferenceParameter &p) {
  float *input_v = p.inputs[0];
  float *template_v = p.inputs[1];
  float *output_v = p.outputs[0];
  auto mode = getMode().str();
  auto input_shape = module::getShape(getInput());
  auto template_shape = module::getShape(getMatch());

  assert(mode == "TM_CCOEFF_NORMED" || mode == "TM_SQDIFF");
  assert(input_shape.size() == 2);
  assert(template_shape.size() == 2);
  assert(input_shape[0] >= template_shape[0]);
  assert(input_shape[1] >= template_shape[1]);

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
      // tmean = BF16(tmean + BF16(template_v[i]));
      tmean += BF16(template_v[i]);
    }
    tmean = BF16(BF16(tmean) / match_size);
  }

  // #pragma omp parallel for schedule(static, omp_schedule(outer_size))
  for (int i = 0; i < outer_size; i++) {
    uint32_t ioffset = i / c * stride + i % c;
    auto input = input_v + ioffset;
    if (mode == "TM_CCOEFF_NORMED") {
      float_t dividend = 0;
      float_t wndSum2 = 0;
      float_t templSum2 = 0;
      float_t imean = 0;
      for (int32_t i = 0; i < match_size; i++) {
        uint32_t offset = i / w * stride + i % w;
        imean += BF16(input[offset]);
      }
      imean = BF16(BF16(imean) / match_size);
      // imean = BF16(imean * BF16(1. / match_size));
      for (int32_t i = 0; i < match_size; i++) {
        uint32_t offset = i / w * stride + i % w;
        auto inp = BF16(input[offset] - imean);
        auto tpl = BF16(template_v[i] - tmean);
        dividend += BF16(inp * tpl);
        wndSum2 += BF16(inp * inp);
        templSum2 += BF16(tpl * tpl);
      }
      dividend = BF16(dividend);
      wndSum2 = BF16(wndSum2);
      templSum2 = BF16(templSum2);

      bf16_lut_mantissa(&wndSum2, &wndSum2, 1, p.inputs[2], p.inputs[3],
                        "mantissa");
      bf16_lut_mantissa(&templSum2, &templSum2, 1, p.inputs[2], p.inputs[3],
                        "mantissa");
      output_v[i] = BF16(dividend * wndSum2 * templSum2);
    } else if (mode == "TM_SQDIFF") {
      float sum = 0;
      for (int i = 0; i < match_size; i++) {
        uint32_t offset = i / w * stride + i % w;
        auto diff = BF16(input[offset] - template_v[i]);
        sum = BF16(sum + BF16(diff * diff));
      }
      // sum =  BF16(BF16(sum) * BF16(scale)) + 1e-5;
      // bf16_lut_mantissa(&sum, &sum, 1, lut->data(), mantissa_lut->data());
      output_v[i] = sum;
    }
  }
  return success();
}

mlir::Type tpu::MatchTemplateOp::type_verify(uint64_t opd_idx,
                                             TypeCastMode &mode) {
  auto op = getOperation();
  return type_verify_case_type(op, opd_idx,
                               Builder(op).getIntegerType(8, false), mode);
}

bool tpu::MatchTemplateOp::support_multi_core() { return false; }
