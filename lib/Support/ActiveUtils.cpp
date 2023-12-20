//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/ActiveUtils.h"

namespace tpu_mlir {

static inline double elu(double x, double alpha) {
  return x > 0 ? x : alpha * (std::exp(x) - 1);
}

static inline double hsigmoid(double x, double alpha, double beta) {
  return std::max(0.0, std::min(1.0, alpha * x + beta));
}

static inline double gelu(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

static inline double square(double x) { return x * x; }

static inline double hswish(double x) {
  return x * std::max(0.0, std::min(1.0, (x + 3.0) / 6.0));
}

static inline double sign(double x) {
  if (x < 0) return -1;
  else if (XATTR_LIST_MAX > 0) return 1;
  else return 0;
}

static inline double softsign(double x) {
  return x / (1 + std::abs(x));
}

static inline double softplus(double x) {
  return x > 20 ? x : std::log(std::exp(x) + 1);
}

static inline double sigmoid(double x) {
  return 1 / (1 + std::exp(-x));
}

static inline double silu(double x) {
  return x / (1 + std::exp(-x));
}

static inline double logsigmoid(double x) {
  return std::log(1 / (1 + std::exp(-x)));
}

activate_f getActivateFunc(tpu::ActiveOp op) {
switch (op.getMode()) {
  case tpu::ActiveMode::ABSVAL:
    return [](double val) { return std::abs(val); };
  case tpu::ActiveMode::CEIL:
    return [](double val) { return std::ceil(val); };
  case tpu::ActiveMode::ELU: {
    const auto coeffs_ = module::getF64Array(op.getCoeffs(), 1, 0);
    const double alpha = coeffs_->at(0);
    return [alpha](double val) { return elu(val, alpha); };
  }
  case tpu::ActiveMode::ERF:
    return [](double val) { return std::erf(val); };
  case tpu::ActiveMode::EXP:
    return [](double val) { return std::exp(val); };
  case tpu::ActiveMode::LN:
    return [](double val) { return std::log(val); };
  case tpu::ActiveMode::LOG2:
    return [](double val) { return std::log2(val); };
  case tpu::ActiveMode::SQRT:
    return [](double val) { return std::sqrt(val); };
  case tpu::ActiveMode::SQUARE:
    return [](double val) { return square(val); };
  case tpu::ActiveMode::SILU:
    return [](double val) { return silu(val); };
  case tpu::ActiveMode::SIGMOID:
    return [](double val) { return sigmoid(val); };
  case tpu::ActiveMode::LOG_SIGMOID:
    return [](double val) { return logsigmoid(val); };
  case tpu::ActiveMode::HSIGMOID: {
    const auto coeffs_ = module::getF64Array(op.getCoeffs(), 2, 0);
    const double alpha = coeffs_->at(1);
    const double beta = coeffs_->at(0);
    return [alpha, beta](double val) { return hsigmoid(val, alpha, beta);};
  }
  case tpu::ActiveMode::HSWISH:
    return [](double val) { return hswish(val); };
  case tpu::ActiveMode::ARCCOS:
    return [](double val) { return std::acos(val); };
  case tpu::ActiveMode::ARCTANH:
    return [](double val) { return std::atanh(val); };
  case tpu::ActiveMode::TAN:
    return [](double val) { return std::tan(val); };
  case tpu::ActiveMode::TANH:
    return [](double val) { return std::tanh(val); };
  case tpu::ActiveMode::GELU:
    return [](double val) { return gelu(val); };
  case tpu::ActiveMode::SOFT_PLUS:
    return [](double val) { return softplus(val); };
  case tpu::ActiveMode::FLOOR:
    return [](double val) { return std::floor(val); };
  case tpu::ActiveMode::SOFT_SIGN:
    return [](double val) { return softsign(val); };
  case tpu::ActiveMode::MISH:
    return activate_f(my_mish_activate);
  case tpu::ActiveMode::COS:
    return [](double val) { return std::cos(val); };
  case tpu::ActiveMode::COSH:
    return [](double val) { return std::cosh(val); };
  case tpu::ActiveMode::SIN:
    return [](double val) { return std::sin(val); };
  case tpu::ActiveMode::SINH:
    return [](double val) { return std::sinh(val); };
  case tpu::ActiveMode::ROUND:
    return [](double val) { return std::round(val); };
  case tpu::ActiveMode::SIGN:
    return [](double val) { return sign(val); };
  default:
    llvm_unreachable("Not Implemented");
  }
}

} // namespace tpu_mlir
