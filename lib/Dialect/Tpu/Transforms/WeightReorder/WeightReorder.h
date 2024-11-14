//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
template <typename T>
inline std::string getTypeName() {
  return "Unknown";
}

template <>
inline std::string getTypeName<int8_t>() {
  return "int8";
}

template <>
inline std::string getTypeName<Float32Type>() {
  return "float32";
}

template <>
inline std::string getTypeName<Float16Type>() {
  return " FP16";
}

template <>
inline std::string getTypeName<BFloat16Type>() {
  return " BF16";
}

template <>
inline std::string getTypeName<tpu::LSTMOp>() {
  return "tpu::LSTMOp";
}

template <>
inline std::string getTypeName<tpu::A16MatMulOp>() {
  return "tpu::A16MatMulOp";
}

template <>
inline std::string getTypeName<tpu::AttentionOp>() {
  return "tpu::AttentionOp";
}

template <>
inline std::string getTypeName<tpu::Conv2DOp>() {
  return "tpu::Conv2DOp";
}

template <>
inline std::string getTypeName<tpu::Conv3DOp>() {
  return "tpu::Conv3DOp";
}

template <>
inline std::string getTypeName<tpu::DeconvOp>() {
  return "tpu::DeconvOp";
}

template <>
inline std::string getTypeName<tpu::GRUOp>() {
  return "tpu::GRUOp";
}

template <>
inline std::string getTypeName<tpu::MatMulOp>() {
  return "tpu::MatMulOp";
}
// 84
template <>
inline std::string getTypeName<tpu::AddOp>() {
  return "tpu::AddOp";
}

template <>
inline std::string getTypeName<tpu::Deconv3DOp>() {
  return "tpu::Deconv3DOp";
}

template <>
inline std::string getTypeName<tpu::GroupNormOp>() {
  return "tpu::GroupNormOp";
}

template <>
inline std::string getTypeName<tpu::MulOp>() {
  return "tpu::MulOp";
}

template <>
inline std::string getTypeName<tpu::PReluOp>() {
  return "tpu::PReluOp";
}

template <>
inline std::string getTypeName<tpu::ScaleOp>() {
  return "tpu::ScaleOp";
}

template <>
inline std::string getTypeName<tpu::SubOp>() {
  return "tpu::SubOp";
}

namespace bm1684 {
template <typename OpTy, typename T>
class WeightReorder : public OpRewriterPatternEx2<OpTy, T> {
public:
  WeightReorder(mlir::MLIRContext *context)
      : OpRewriterPatternEx2<OpTy, T>(context, generatePatternName()) {}

public:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override;

  bool shouldPrint(OpTy op) const override { return false; }

private:
  std::string generatePatternName() const {
    return std::string("WeightReorder_") + getTypeName<OpTy>() + "_" +
           getTypeName<T>();
  }
};
} // namespace bm1684

namespace bm1684x {
template <typename OpTy, typename T>
class WeightReorder : public OpRewriterPatternEx2<OpTy, T> {
public:
  WeightReorder(mlir::MLIRContext *context)
      : OpRewriterPatternEx2<OpTy, T>(context, generatePatternName()) {}

public:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override;
  bool shouldPrint(OpTy op) const override { return false; }

private:
  std::string generatePatternName() const {
    return std::string("WeightReorder_") + getTypeName<OpTy>() + "_" +
           getTypeName<T>();
  }
};

template <typename OpTy, typename T>
class WeightDeReorder : public OpRewriterPatternEx2<OpTy, T> {
public:
  WeightDeReorder(mlir::MLIRContext *context)
      : OpRewriterPatternEx2<OpTy, T>(context, generatePatternName()) {}

public:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override;
  bool shouldPrint(OpTy op) const override { return false; }

private:
  std::string generatePatternName() const {
    return std::string("WeightReorder_") + getTypeName<OpTy>() + "_" +
           getTypeName<T>();
  }
};

} // namespace bm1684x

namespace cv18xx {
template <typename OpTy, typename T>
class WeightReorder : public OpRewriterPatternEx2<OpTy, T> {
public:
  WeightReorder(mlir::MLIRContext *context)
      : OpRewriterPatternEx2<OpTy, T>(context, generatePatternName()) {}

public:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override;
  bool shouldPrint(OpTy op) const override { return false; }

private:
  std::string generatePatternName() const {
    return std::string("WeightReorder_") + getTypeName<OpTy>() + "_" +
           getTypeName<T>();
  }
};
} // namespace cv18xx

} // namespace tpu_mlir
