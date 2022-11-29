//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define SIGMOID_BF16_LUT_RANGE 12
#define TANH_BF16_LUT_RANGE 15
#define EXP_BF16_LUT_RANGE 15
#define ELU_BF16_LUT_RANGE 15
#define MISH_BF16_LUT_RANGE 8
#define SOFTPLUS_BF16_LUT_RANGE 8
#define SWISH_BF16_LUT_RANGE 12
#define LOG_BF16_LUT_RANGE 8
#define GELU_BF16_LUT_RANGE 8

using namespace llvm;
using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

namespace tpu_mlir {

// create lookup table
using activate_f = std::function<double(double)>;

Value create_lookup_table(Value in, Value out, bool asymmetric,
                          activate_f &&func);

Value create_lookup_table(Operation *owner, const std::vector<float> &table);
Value create_lookup_table(Operation *owner, const std::vector<int> &table);

void bf16_gen_base_slope_table(float *base_table, float *slope_table,
                               float range_start, float range_end,
                               activate_f &&func);

void bf16_lut_slope(float *input, float *output, int size, float *base_table,
                    float *slope_table, float range_start, float range_end);

void bf16_gen_exponent_mantissa_table(const std::string &name, float *exp_table,
                                      float *mantissa_table, float param0, float param1);

void bf16_lut_mantissa(float *input, float *output, int size, float *exp_table,
                       float *mantissa_table, const std::string &method);
} // namespace tpu_mlir
