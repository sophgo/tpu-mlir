//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

static inline std::vector<int64_t> string2vec(std::string slist) {
  std::vector<int64_t> outvec;
  std::string idx_str = "";
  for (auto s : slist) {
    int idx;
    if (s == ',') {
      idx = atoi(idx_str.c_str());
      idx_str = "";
      outvec.push_back(idx);
    } else {
      idx_str += s;
    }
  }
  if (idx_str.size())
    outvec.push_back(atoi(idx_str.c_str()));
  return outvec;
}

class LoraUpdatePattern : public OpRewriterPatternEx<tpu::A16MatMulOp> {
public:
  LoraUpdatePattern(mlir::MLIRContext *context, int64_t rank,
                    std::vector<int64_t> weight_idx)
      : OpRewriterPatternEx<tpu::A16MatMulOp>(context, "LoraUpdatePattern"),
        rank(rank), weight_idx(weight_idx) {}

  template <typename TyOp>
  tpu::MatMulOp createMatMulOp(TyOp &op, std::string suffix, int64_t out_dim,
                               PatternRewriter &rewriter) const {
    Value in = op.getInput();
    Value out = op.getOutput();
    auto in_shape = module::getShapeVec(in);
    auto out_shape = module::getShapeVec(out);
    auto weight_op = op.getOperand(1).template getDefiningOp<top::WeightOp>();
    std::vector<NamedAttribute> attrs;

    Value in_mm;
    std::vector<int64_t> weight_shape = {0, out_dim};
    if (out_dim == rank) {
      in_mm = op.getInput();
      weight_shape[0] = in_shape.back();
    } else {
      in_mm = op.getOutput();
      weight_shape[0] = out_shape.back();
    }

    int element_nums = std::accumulate(weight_shape.begin(), weight_shape.end(),
                                       1, std::multiplies<int64_t>());

    auto weight_type =
        RankedTensorType::get(weight_shape, module::getElementType(in));
    auto zero_data = std::make_shared<std::vector<uint16_t>>(element_nums, 0);
    auto in_weight = top::WeightOp::create<uint16_t>(weight_op, suffix,
                                                     *zero_data, weight_type);

    auto none = module::getNoneOp(op);
    std::vector<Value> operands = {in_mm, in_weight, none, none, none};

    auto mm_loc = module::getLocLike(out, suffix);

    out_shape[out_shape.size() - 1] = out_dim;
    auto mm_type = module::getTypeLike(out, out_shape);
    auto new_mm_op =
        rewriter.create<tpu::MatMulOp>(mm_loc, mm_type, operands, attrs);

    return new_mm_op;
  }

  LogicalResult matchAndRewriteImpl(tpu::A16MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    // Check if the operation name matches any in the weight_idx, if provided
    if (weight_idx.size() != 0) {
      // auto op_name = op.getOperation()->getName().getStringRef().str();
      // if (std::find(weight_idx.begin(), weight_idx.end(), op_name) ==
      // weight_idx.end()) {
      //   return failure();
      // }
    }
    if (rank == 0) {
      return failure();
    }

    // Get the input and weights of the A16MatMulOp
    Value out = op.getOutput();
    auto out_shape = module::getShapeVec(out);
    std::vector<Operation *> users(op->user_begin(), op->user_end());
    auto mm_op_0 =
        createMatMulOp<tpu::A16MatMulOp>(op, "lora_0", rank, rewriter);
    auto mm_op_1 = createMatMulOp<tpu::MatMulOp>(mm_op_0, "lora_1",
                                                 out_shape.back(), rewriter);

    rewriter.setInsertionPointAfter(op);
    auto add_loc = module::getLocLike(out, "lora_add");
    auto add_type = module::getTypeLike(out, out_shape);
    auto add_op = rewriter.create<AddOp>(add_loc, add_type,
                                         ValueRange{out, mm_op_1.getOutput()});

    for (auto *user : users) {
      for (unsigned i = 0; i < user->getNumOperands(); ++i) {
        if (out == user->getOperand(i)) {
          user->setOperand(i, add_op.getResult());
        }
      }
    }

    return success();
  };

private:
  int64_t rank;
  std::vector<int64_t> weight_idx;
};

class FutureUpdatePass : public FutureUpdateBase<FutureUpdatePass> {
public:
  FutureUpdatePass() {}
  void runOnOperation() override {
    auto ctx = &getContext();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      for (auto func : s.getOps<FuncOp>()) {
        RewritePatternSet patterns(ctx);

        std::vector<int64_t> weight_idx = string2vec(weight_list);
        patterns.add<LoraUpdatePattern>(ctx, rank, weight_idx);

        auto config = GreedyRewriteConfig();
        config.maxIterations = 1; // apply each pattern only once.
        applyPatternsAndFoldGreedily(func, std::move(patterns), config);
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createFutureUpdatePass() {
  return std::make_unique<FutureUpdatePass>();
}
} // namespace tpu
} // namespace tpu_mlir
