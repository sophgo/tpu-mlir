#pragma once
#include "tpu_mlir/Support/Module.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/PatternMatch.h>
#include <mutex>
#include <string>
#include <unordered_map>

template <typename SourceOp>
class OpRewriterPatternEx : public mlir::OpRewritePattern<SourceOp> {
public:
  OpRewriterPatternEx(mlir::MLIRContext *context,
                      llvm::StringRef patternName = "",
                      mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<SourceOp>(context, benefit),
        patternName(patternName) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::LogicalResult result = matchAndRewriteImpl(op, rewriter);
    if (mlir::succeeded(result)) {
      if (!patternName.empty()) {
        std::lock_guard<std::mutex> lock(
            tpu_mlir::module::patternMatchCountsMutex);
        ++tpu_mlir::module::patternMatchCounts[patternName];
      }
      if (shouldPrint(op) && !patternName.empty()) {
        //  #todo : print opname,no save mode has bug need to solve,this is a
        //  temporary solution
        PASS_LOG_DEBUG_BLOCK({
          llvm::outs() << patternName << " : " << op.getOperationName()
                       << " succeed!";
        });
      }
    }
    return result;
  }

protected:
  virtual mlir::LogicalResult
  matchAndRewriteImpl(SourceOp op, mlir::PatternRewriter &rewriter) const = 0;

  virtual bool shouldPrint(SourceOp op) const { return true; }

private:
  std::string patternName;
  static void printPatternMatchCounts() {
    std::lock_guard<std::mutex> lock(tpu_mlir::module::patternMatchCountsMutex);
    for (const auto &entry : tpu_mlir::module::patternMatchCounts) {
      std::cout << "Pattern [" << entry.first << "] matched " << entry.second
                << " times.\n";
    }
  }
};

template <typename SourceOp, typename ElementType>
class OpRewriterPatternEx2 : public mlir::OpRewritePattern<SourceOp> {
public:
  OpRewriterPatternEx2(mlir::MLIRContext *context,
                       llvm::StringRef patternName = "",
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<SourceOp>(context, benefit),
        patternName(patternName.str()) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::PatternRewriter &rewriter) const override {
    mlir::LogicalResult result = matchAndRewriteImpl(op, rewriter);
    if (mlir::succeeded(result)) {
      if (shouldPrint(op) && !patternName.empty()) {
        PASS_LOG_DEBUG_BLOCK({
          llvm::outs() << patternName << "_" << op.getOperationName()
                       << " succeed!" << "\n";
        });
      }
    }
    return result;
  }

protected:
  virtual mlir::LogicalResult
  matchAndRewriteImpl(SourceOp op, mlir::PatternRewriter &rewriter) const = 0;

  virtual bool shouldPrint(SourceOp op) const { return true; }

private:
  std::string patternName;
};

class OpRewriterPatternEx3 : public mlir::RewritePattern {
public:
public:
  OpRewriterPatternEx3(mlir::MLIRContext *context,
                       llvm::StringRef patternName = "",
                       mlir::PatternBenefit benefit = 1)
      : mlir::RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), benefit,
                             context),
        patternName(patternName.str()) {}

  OpRewriterPatternEx3(mlir::MLIRContext *context, llvm::StringRef patternName,
                       mlir::PatternBenefit benefit, llvm::StringRef typeName)
      : mlir::RewritePattern(typeName, benefit, context),
        patternName(patternName.str()) {}
  mlir::LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::LogicalResult result = matchAndRewriteImpl(op, rewriter);
    if (mlir::succeeded(result)) {
      if (shouldPrint(op) && !patternName.empty()) {
        PASS_LOG_DEBUG_BLOCK({
          llvm::outs() << patternName << "_" << op->getName().getStringRef()
                       << " succeed!" << "\n";
        });
      }
    }
    return result;
  }

protected:
  virtual mlir::LogicalResult
  matchAndRewriteImpl(Operation *op, mlir::PatternRewriter &rewriter) const = 0;

  virtual bool shouldPrint(Operation *op) const { return true; }

private:
  std::string patternName;
};

#include "mlir/Transforms/DialectConversion.h"
class ConversionPatternEx : public ConversionPattern {
public:
  ConversionPatternEx(TypeConverter &typeConverter, StringRef patternName,
                      PatternBenefit benefit, MLIRContext *ctx)
      : ConversionPattern(typeConverter, patternName, benefit, ctx),
        patternName(patternName.str()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LogicalResult result = matchAndRewriteImpl(op, operands, rewriter);
    if (succeeded(result)) {
      if (shouldPrint(op) && !patternName.empty()) {
        PASS_LOG_DEBUG_BLOCK({
          llvm::outs() << "Pattern [" << patternName
                       << "] successfully applied to operation: "
                       << op->getName().getStringRef() << "\n";
        });
      }
    }
    return result;
  }

protected:
  virtual LogicalResult
  matchAndRewriteImpl(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter) const = 0;

  virtual bool shouldPrint(Operation *op) const { return true; }

private:
  std::string patternName;
};
