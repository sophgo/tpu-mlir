//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/LutFunc.h"

static void lrn_inference_cv18xx(InferenceParameter &p, int64_t size,
                                 double alpha, double bias, int64_t n,
                                 int64_t c, int64_t h, int64_t w) {
  float alpha_bf16 = BF16(static_cast<float>(alpha / size));
  float bias_bf16 = BF16(static_cast<float>(bias));
  int64_t move_counts = (size - 1) / 2;
  std::vector<float> elem(n * c * h * w);
// y = x * (k + sum(ax^2))^(-beta)
// step 1: elem = ax ^ 2
#pragma omp parallel for schedule(static, omp_schedule(n *c))
  for (int64_t i = 0; i < n * c; ++i) {
    for (int64_t j = 0; j < h * w; ++j) {
      elem[i * h * w + j] =
          bf16_mul(alpha_bf16, bf16_mul(p.inputs[0][i * h * w + j],
                                        p.inputs[0][i * h * w + j]));
    }
  }

  // step 2: tmp = sum(elem)
  std::vector<float> tmp(n * c * h * w, BF16(0));
  for (int64_t _n = 0; _n < n; ++_n) {
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int64_t _c = 0; _c < c; ++_c) {
      for (int64_t i = 0; i < h * w; ++i) {
        float tmp_value = elem[_n * c * h * w + _c * h * w + i];
        for (int64_t step = 1; step <= move_counts && step; ++step) {
          int64_t left_shift_c = _c + step;
          if (left_shift_c < c) {
            tmp_value = bf16_add(
                tmp_value, elem[_n * c * h * w + left_shift_c * h * w + i]);
          }
          int64_t right_shift_c = _c - step;
          if (right_shift_c >= 0) {
            tmp_value = bf16_add(
                tmp_value, elem[_n * c * h * w + right_shift_c * h * w + i]);
          }
        }
        tmp[_n * c * h * w + _c * h * w + i] = tmp_value;
      }
    }
  }

  // step 3: elem = k + tmp
#pragma omp parallel for schedule(static, omp_schedule(n *c))
  for (int64_t i = 0; i < n * c; ++i) {
    for (int64_t j = 0; j < h * w; ++j) {
      elem[i * h * w + j] = bf16_add(tmp[i * h * w + j], bias_bf16);
    }
  }

  // step 4: tmp = elem ^ (-beta)
  bf16_lut_mantissa(elem.data(), tmp.data(), n * c * h * w, p.inputs[1],
                    p.inputs[2], "mantissa");

  // step 5: output = x * tmp
#pragma omp parallel for schedule(static, omp_schedule(n *c))
  for (int64_t i = 0; i < n * c; ++i) {
    for (int64_t j = 0; j < h * w; ++j) {
      p.outputs[0][i * h * w + j] =
          bf16_mul(tmp[i * h * w + j], p.inputs[0][i * h * w + j]);
    }
  }
}

LogicalResult tpu::LRNOp::init(InferenceParameter &p) {
  auto alpha_ = getAlpha().convertToDouble();
  auto beta_ = getBeta().convertToDouble();
  auto bias_ = getBias().convertToDouble();
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isBF16()) {
    alpha_ = BF16(alpha_);
    beta_ = BF16(beta_);
    bias_ = BF16(bias_);
  } else if (out_type.isF16()) {
    alpha_ = F16(alpha_);
    beta_ = F16(beta_);
    bias_ = F16(bias_);
  }

  auto lrn = new LRN();
  (*lrn)
      .src(p.inputs[0], module::getShape(getInput()))
      .dst(p.outputs[0], module::getShape(getOutput()))
      .size(getSize())
      .param(alpha_, beta_, bias_)
      .algorithem(algorithm::lrn_across_channels)
      .setup();

  p.handle = (void *)lrn;
  return success();
}

void tpu::LRNOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto lrn = (LRN *)p.handle;
    delete lrn;
    p.handle = nullptr;
  }
}

LogicalResult tpu::LRNOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  if (out_type.isa<FloatType>()) {
    if (module::isCV18xx()) {
      int64_t n, c, h, w;
      module::getNCHW(getInput(), n, c, h, w);
      lrn_inference_cv18xx(p, getSize(), getAlpha().convertToDouble(),
                           getBias().convertToDouble(), n, c, h, w);
    } else {
      auto lrn = (LRN *)p.handle;
      lrn->run();
    }

    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }

  } else if (out_type.isInteger(8)) {
    auto lrn = (LRN *)p.handle;
    lrn->run();
  } else {
    dump();
    llvm_unreachable("not support type");
  }

  return success();
}

LogicalResult tpu::LRNOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return success();
  }
  return failure();
}

bool tpu::LRNOp::support_multi_core() { return false; }
