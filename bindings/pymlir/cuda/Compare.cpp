//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

namespace {

const std::map<std::string, int> mode_map = {
    {"Equal", 0},          {"Greater", 1},        {"GreaterOrEqual", 2},
    {"Less", 3},           {"LessOrEqual", 4},    {"NotEqual", 5},
    {"And", 6},            {"Not", 7},            {"Xor", 8}};

int getModeInt(const std::string &mode_str) {
  auto it = mode_map.find(mode_str);
  return (it != mode_map.end()) ? it->second : -1;
}

} // namespace

void py_cuda::cudaCompareOp(top::CompareOp op) {
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto out = op.getOutput();

  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(lhs, n0, c0, h0, w0, false);
  module::getNCHW(rhs, n1, c1, h1, w1, false);
  module::getNCHW(out, n2, c2, h2, w2, false);

  cuda::bmCompare4DF32(getCudaData(lhs), getCudaData(rhs), getCudaData(out),
                        getModeInt(op.getModeAttr().str()),
                        n0, c0, h0, w0,
                        n1, c1, h1, w1,
                        n2, c2, h2, w2);
}

void py_cuda::cudaCompareOp(tpu::CompareOp op) {
  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto out = op.getOutput();

  int64_t n0, c0, h0, w0, n1, c1, h1, w1, n2, c2, h2, w2;
  module::getNCHW(lhs, n0, c0, h0, w0, false);
  module::getNCHW(rhs, n1, c1, h1, w1, false);
  module::getNCHW(out, n2, c2, h2, w2, false);

  cuda::bmCompare4DF32(getCudaData(lhs), getCudaData(rhs), getCudaData(out),
                        getModeInt(op.getModeAttr().str()),
                        n0, c0, h0, w0,
                        n1, c1, h1, w1,
                        n2, c2, h2, w2);
}

// ==========================================================================
// CompareConst
// ==========================================================================

void py_cuda::cudaCompareConstOp(top::CompareConstOp op) {
  float const_v = op.getConstVal().convertToDouble();
  int mode = getModeInt(op.getModeAttr().str());
  bool inversed = op.getInversed();

  int64_t n, c, h, w;
  module::getNCHW(op.getOutput(), n, c, h, w, false);

  cuda::bmCompareConst4DF32(getCudaData(op.getInput()), const_v,
                             getCudaData(op.getOutput()),
                             mode, inversed, n, c, h, w);
}

void py_cuda::cudaCompareConstOp(tpu::CompareConstOp op) {
  float const_v = op.getConstVal().convertToDouble();
  int mode = getModeInt(op.getModeAttr().str());
  bool inversed = op.getInversed();

  int64_t n, c, h, w;
  module::getNCHW(op.getOutput(), n, c, h, w, false);

  cuda::bmCompareConst4DF32(getCudaData(op.getInput()), const_v,
                             getCudaData(op.getOutput()),
                             mode, inversed, n, c, h, w);
}
