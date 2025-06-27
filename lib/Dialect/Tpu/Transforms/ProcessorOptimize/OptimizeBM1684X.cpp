//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/DistributeUtils.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/LutFunc.h"

using namespace llvm;
using namespace tpu_mlir::backend;
namespace tpu_mlir {

namespace bm1684x {
class MatMulHdimBatchPattern : public OpRewriterPatternEx<tpu::MatMulOp> {
  // Case1: Permute -> MatMul <- Permute
  // Case2: Reshape -> MatMul <- Permute
  // Case3: Left    -> MatMul <- Permute
  // Case4: Permute -> MatMul <- Tile <- Permute
public:
  MatMulHdimBatchPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMulHdimBatchPattern",
                                           benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::MatMulOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto userIt = op.getOutput().user_begin();
    if (userIt != op.getOutput().user_end()) {
      auto _nextOp_ = *userIt;
      if (_nextOp_ && isa<tpu::RequantIntAxisOp>(
                          _nextOp_)) { // First do MatMulRequantIntFusion and
                                       // then do MatMulHdimBatchPattern
        auto RequantIntAxis_Op = dyn_cast<tpu::RequantIntAxisOp>(_nextOp_);
        auto Fuse_Rq_Axis = RequantIntAxis_Op.getFuseRqAxis();
        if (Fuse_Rq_Axis)
          return failure();
      }
    }
    int flag_swap = 0;

    //  if (module::isAsymmetric()) {
    //    return failure();
    //  }

    // 1. Define Left and Right
    auto left = op.getInput();
    auto right = op.getRight();
    auto stype = module::getStorageType(left);
    auto hdim_is_batch = op.getHdimIsBatch();
    if (stype.isF32() || hdim_is_batch) {
      return failure();
    }

    // 2. Check Left and Right
    auto l_is_weight = module::isWeight(left);
    auto r_is_weight = module::isWeight(right);
    if (l_is_weight && r_is_weight) {
      return failure();
    }
    auto l_op = left.getDefiningOp();
    auto r_op = right.getDefiningOp();
    if (!isa<tpu::PermuteOp>(l_op) && !isa<tpu::PermuteOp>(r_op)) {
      return failure();
    }
    // eliminate Tile
    if (isa<tpu::TileOp>(r_op)) {
      auto tile_op = dyn_cast<tpu::TileOp>(r_op);
      if (!isa<tpu::PermuteOp>(tile_op.getOperand(0).getDefiningOp())) {
        return failure();
      }
      auto permute_op =
          dyn_cast<tpu::PermuteOp>(tile_op.getOperand(0).getDefiningOp());
      rewriter.replaceAllUsesWith(tile_op->getResult(0),
                                  permute_op->getResult(0));
      rewriter.eraseOp(tile_op);
      r_op = permute_op;
    }

    // 3. Convert MatMul to HdimBatch MatMul
    if (!l_is_weight && !r_is_weight) {
      // When Left and Right is Tensor

      auto l_output_shape = module::getShape(l_op->getResult(0));
      auto r_output_shape = module::getShape(r_op->getResult(0));
      if (l_output_shape.size() < 3 || r_output_shape.size() < 3) {
        return failure();
      }
      // Swap Left and Right
      if (isa<tpu::PermuteOp>(l_op) && !isa<tpu::PermuteOp>(r_op) &&
          l_output_shape[2] == r_output_shape[2]) {
        std::swap(l_op, r_op);
        flag_swap = 1;
      }

      if (isa<tpu::PermuteOp>(l_op) && isa<tpu::PermuteOp>(r_op)) {
        // Case1
        // Left  -> Permute -\              Left  -\
        //                   ->  MatMul ->         -> MatMul
        // Right -> Permute -/              Right -/
        auto l_trans_op = dyn_cast<tpu::PermuteOp>(l_op);
        auto r_trans_op = dyn_cast<tpu::PermuteOp>(r_op);
        if (!l_trans_op->hasOneUse() || !r_trans_op->hasOneUse()) {
          return failure();
        }
        auto l_order = module::getI64Array(l_trans_op.getOrder());
        auto r_order = module::getI64Array(r_trans_op.getOrder());
        if (false == (l_order->size() == 4 && l_order->at(0) == 0 &&
                      l_order->at(1) == 2 && r_order->size() == 4 &&
                      r_order->at(0) == 0 && r_order->at(1) == 2)) {
          return failure();
        }
        auto l_trans = op.getLeftTranspose();
        auto r_trans = op.getRightTranspose();
        if (l_order->at(2) == 3 && l_order->at(3) == 1) {
          l_trans = !l_trans;
        }
        if (r_order->at(2) == 3 && r_order->at(3) == 1) {
          r_trans = !r_trans;
        }
        if (l_trans == true && r_trans == false) {
          // mm2 not support l_trans && !r_trans
          return failure();
        }
        op->setAttr("hdim_is_batch", rewriter.getBoolAttr(!hdim_is_batch));
        op->setAttr("left_transpose", rewriter.getBoolAttr(l_trans));
        op->setAttr("right_transpose", rewriter.getBoolAttr(r_trans));
        op->setOperand(0, l_trans_op.getInput());
        op->setOperand(1, r_trans_op.getInput());
        rewriter.eraseOp(l_trans_op);
        rewriter.eraseOp(r_trans_op);
      } else if (isa<tpu::ReshapeOp>(l_op) && isa<tpu::PermuteOp>(r_op)) {
        // Case2
        // Left  -> Reshape -\              Left(+ Reshape)-\
        //                   ->  MatMul ->                  -> MatMul
        // Right -> Permute -/              Right          -/
        auto l_trans_op = dyn_cast<tpu::ReshapeOp>(l_op);
        auto r_trans_op = dyn_cast<tpu::PermuteOp>(r_op);
        if (!l_trans_op->hasOneUse() || !r_trans_op->hasOneUse()) {
          return failure();
        }

        auto r_order = module::getI64Array(r_trans_op.getOrder());
        auto r_shape = module::getShape(r_trans_op.getOutput());
        auto r_in_shape = module::getShape(r_trans_op.getInput());
        auto l_in_shape = module::getShape(l_trans_op.getInput());
        auto l_out_shape = module::getShape(l_trans_op.getOutput());
        if (false == (r_order->size() == 4 && r_order->at(0) == 0 &&
                      r_order->at(1) == 2 && l_out_shape[1] == r_shape[1] &&
                      l_in_shape[1] == l_out_shape[2])) {
          return failure();
        }

        auto l_trans = op.getLeftTranspose();
        auto r_trans = op.getRightTranspose();
        if (r_order->at(2) == 3 && r_order->at(3) == 1) {
          r_trans = !r_trans;
        }

        // Check Shape (left.shape[-1] == right.shape[-2])
        bool remove_reshape = l_in_shape.size() == l_out_shape.size();
        if (!(l_in_shape.size() >= 2 && r_in_shape.size() >= 2))
          return failure();
        int l_K_dim = l_in_shape.size() - 1 - l_trans - l_trans * hdim_is_batch;
        int r_K_dim = r_in_shape.size() - 2 + r_trans +
                      r_trans * hdim_is_batch - hdim_is_batch;
        if (l_in_shape[l_K_dim] != r_in_shape[r_K_dim]) {
          if (l_out_shape.size() == 4 && l_out_shape[2] == 1) {
            std::vector<int64_t> new_l_shape = l_out_shape;
            new_l_shape[2] = l_out_shape[1];
            new_l_shape[1] = 1;
            module::setShape(l_trans_op.getOutput(), new_l_shape);
            l_trans_op.getOutput().dump();
            remove_reshape = false;
            l_out_shape = module::getShape(l_trans_op.getOutput());
          } else {
            return failure();
          }
        }

        if (!hdim_is_batch && l_in_shape.size() > 2 && r_in_shape.size() > 2) {
          int min_len = std::min(remove_reshape * l_in_shape.size() +
                                     (1 - remove_reshape) * l_out_shape.size(),
                                 r_in_shape.size());
          for (int i = 0; i < min_len - 2; i++) {
            int ls;
            if (remove_reshape)
              ls = l_in_shape[l_in_shape.size() - 3 - i];
            else
              ls = l_out_shape[l_out_shape.size() - 3 - i];
            int rs = r_in_shape[r_in_shape.size() - 3 - i];
            if (!(ls == rs || ls == 1 || rs == 1)) {
              return failure();
            }
          }
        }

        // Define Param
        op->setAttr("hdim_is_batch", rewriter.getBoolAttr(!hdim_is_batch));
        op->setAttr("left_transpose", rewriter.getBoolAttr(false));
        op->setAttr("right_transpose", rewriter.getBoolAttr(r_trans));
        if (remove_reshape) {
          op->setOperand(0, l_trans_op.getInput());
          rewriter.eraseOp(l_trans_op);
        }
        op->setOperand(1, r_trans_op.getInput());
        rewriter.eraseOp(r_trans_op);
      } else if (!isa<tpu::PermuteOp>(l_op) && isa<tpu::PermuteOp>(r_op)) {
        // Case3
        // Left  ->         -\              Left  Permute -\
        //                   ->  MatMul ->                -> MatMul
        // Right -> Permute -/              Right         -/
        auto l_trans_op = l_op;
        auto r_trans_op = dyn_cast<tpu::PermuteOp>(r_op);
        if (!l_trans_op->hasOneUse() || !r_trans_op->hasOneUse()) {
          return failure();
        }

        auto r_order = module::getI64Array(r_trans_op.getOrder());
        auto r_shape = module::getShape(r_trans_op.getOutput());
        auto l_shape = module::getShape(l_trans_op->getResult(0));
        if (false == (r_order->size() == 4 && r_order->at(0) == 0 &&
                      r_order->at(1) == 2 && l_shape[1] == r_shape[1])) {
          return failure();
        }
        auto op_name = module::getName(l_op->getResult(0)).str();
        // Add ReshapeOp or PermuteOp
        Operation *new_l_trans_op;

        std::vector<NamedAttribute> attrs;
        std::vector<int64_t> out_order = {0, 2, 1, 3};
        auto l_trans_type = RankedTensorType::get(
            {l_shape[0], l_shape[2], l_shape[1], l_shape[3]},
            module::getElementType(left));
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr(out_order)));
        new_l_trans_op = rewriter.create<tpu::PermuteOp>(
            NameLoc::get(rewriter.getStringAttr(op_name + "_permute")),
            l_trans_type,
            ValueRange{l_trans_op->getResult(0), module::getNoneOp(op)}, attrs);
        auto r_trans = op.getRightTranspose();
        if (r_order->at(2) == 3 && r_order->at(3) == 1) {
          r_trans = !r_trans;
        }

        // Define Param
        op->setAttr("hdim_is_batch", rewriter.getBoolAttr(!hdim_is_batch));
        op->setAttr("left_transpose", rewriter.getBoolAttr(false));
        op->setAttr("right_transpose", rewriter.getBoolAttr(r_trans));
        op->setOperand(0, new_l_trans_op->getResult(0));
        op->setOperand(1, r_trans_op.getInput());
        rewriter.eraseOp(r_trans_op);
      } else {
        return failure();
      }
    } else if (l_is_weight || r_is_weight) {
      // When Left or Right is weight
      auto trans_op = r_is_weight
                          ? dyn_cast<tpu::PermuteOp>(left.getDefiningOp())
                          : dyn_cast<tpu::PermuteOp>(right.getDefiningOp());
      auto weight_op = l_is_weight ? left.getDefiningOp<top::WeightOp>()
                                   : right.getDefiningOp<top::WeightOp>();
      if (!weight_op->hasOneUse()) {
        return failure();
      }
      if (!(trans_op && trans_op->hasOneUse())) {
        return failure();
      }

      auto order = module::getI64Array(trans_op.getOrder());
      if (false ==
          (order->size() == 4 && order->at(0) == 0 && order->at(1) == 2)) {
        return failure();
      }
      auto l_trans = op.getLeftTranspose();
      auto r_trans = op.getRightTranspose();
      if (r_is_weight && order->at(2) == 3 && order->at(3) == 1) {
        l_trans = !l_trans;
      }
      if (l_is_weight && order->at(2) == 3 && order->at(3) == 1) {
        r_trans = !r_trans;
      }
      if (l_trans == true && r_trans == false) {
        // mm2 not support l_trans && !r_trans
        return failure();
      }

      // transpose the weight
      auto weight_type = module::getElementType(weight_op.getOutput());
      auto weight_shape = module::getShape(weight_op.getOutput());
      if (weight_shape.size() != 4) {
        return failure();
      }
      if (weight_type.isInteger(8)) {
        auto weight_data = weight_op.read<uint8_t>();
        auto weight_trans =
            std::make_shared<std::vector<uint8_t>>(weight_data->size(), 0);
        function_permute(weight_data->data(), weight_trans->data(),
                         weight_shape, {0, 2, 1, 3});
        std::vector<int64_t> weight_new_shape = {
            weight_shape[0], weight_shape[2], weight_shape[1], weight_shape[3]};
        rewriter.setInsertionPointAfter(op);
        auto type = RankedTensorType::get(weight_new_shape, weight_type);
        auto new_weight = top::WeightOp::create<uint8_t>(op, "transposed",
                                                         *weight_trans, type);
        op->setOperand(0, l_is_weight ? new_weight : trans_op.getInput());
        op->setOperand(1, r_is_weight ? new_weight : trans_op.getInput());
      } else if (weight_type.isF16() || weight_type.isBF16()) {
        auto weight_data = weight_op.read<uint16_t>();
        auto weight_trans =
            std::make_shared<std::vector<uint16_t>>(weight_data->size(), 0);
        function_permute(weight_data->data(), weight_trans->data(),
                         weight_shape, {0, 2, 1, 3});
        std::vector<int64_t> weight_new_shape = {
            weight_shape[0], weight_shape[2], weight_shape[1], weight_shape[3]};
        rewriter.setInsertionPointAfter(op);
        auto type = RankedTensorType::get(weight_new_shape, weight_type);
        auto new_weight = top::WeightOp::create<uint16_t>(op, "transposed",
                                                          *weight_trans, type);
        op->setOperand(0, l_is_weight ? new_weight : trans_op.getInput());
        op->setOperand(1, r_is_weight ? new_weight : trans_op.getInput());
      } else {
        llvm_unreachable("Weight type error!");
      }

      op->setAttr("hdim_is_batch", rewriter.getBoolAttr(!hdim_is_batch));
      op->setAttr("left_transpose", rewriter.getBoolAttr(l_trans));
      op->setAttr("right_transpose", rewriter.getBoolAttr(r_trans));

      rewriter.eraseOp(trans_op);
      rewriter.eraseOp(weight_op);
    } else {
      return failure();
    }

    // 4. Modify matmul out shape and name
    auto mat_out = op->getResult(0);
    auto trans_type = mat_out.getType();
    auto out_shape = module::getShape(mat_out);
    std::vector<int64_t> new_out_shape(4, 0);
    new_out_shape[0] = out_shape[0];
    new_out_shape[1] = out_shape[2];
    new_out_shape[2] = out_shape[1];
    new_out_shape[3] = out_shape[3];
    module::setShape(mat_out, new_out_shape);
    auto ori_loc = op->getLoc();
    module::setLocSuffix(op, "hdim_is_batch");

    // 5. Add Transpose(0,2,1,3) to output
    rewriter.setInsertionPointAfter(op);
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> out_order = {0, 2, 1, 3};
    if (flag_swap && !hdim_is_batch) {
      out_order[2] = 3;
      out_order[3] = 1;
    }
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(out_order)));
    auto trans_op = rewriter.create<tpu::PermuteOp>(
        ori_loc, trans_type, ValueRange{mat_out, module::getNoneOp(op)}, attrs);
    rewriter.replaceAllUsesExcept(mat_out, trans_op->getResult(0), trans_op);
    return success();
  }
};

class MatMulLeftReusePattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  MatMulLeftReusePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMulLeftReusePattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (in_op->hasOneUse()) {
      op.setLeftReuse(0);
    } else {
      op.setLeftReuse(1);
    }
    return failure();
  }
};

/*
  Do:
    Reshape
            + MatMul -->>  MatMul
    Reshape

  When:
      Reshape (1,N,K) -> (1,1,N,K) or (1,N,K) -> (1,N,1,K)
*/
class MatMulRemoveReshapePattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  MatMulRemoveReshapePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(
            context, "MatMulRemoveReshapePattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto left_op =
        dyn_cast_or_null<tpu::ReshapeOp>(op.getInput().getDefiningOp());
    auto right_op =
        dyn_cast_or_null<tpu::ReshapeOp>(op.getRight().getDefiningOp());
    if (!(left_op && left_op->hasOneUse()))
      return failure();
    if (!(right_op && right_op->hasOneUse()))
      return failure();

    if (module::getShape(left_op.getInput()).size() !=
        module::getShape(right_op.getInput()).size())
      return failure();

    if (module::getShape(left_op.getInput()).size() <= 2) {
      return failure();
    }

    auto reshape_is_unsqueeze = [](tpu::ReshapeOp reshape_op) {
      std::vector<int64_t> in_shape = module::getShape(reshape_op.getInput());
      std::vector<int64_t> out_shape = module::getShape(reshape_op.getOutput());
      std::vector<int64_t> in_set;
      for (auto in : in_shape) {
        if (in != 1)
          in_set.emplace_back(in);
      }
      std::vector<int64_t> out_set;
      for (auto out : out_shape) {
        if (out != 1)
          out_set.emplace_back(out);
      }
      return (out_shape.size() > in_shape.size() && in_set == out_set);
    };

    if (!reshape_is_unsqueeze(left_op) || !reshape_is_unsqueeze(right_op))
      return failure();

    op.setOperand(0, left_op.getInput());
    op.setOperand(1, right_op.getInput());
    rewriter.eraseOp(left_op);
    rewriter.eraseOp(right_op);
    return success();
  }
};

/**
 * Improve uArchRate of matmul: significant perf benefit in f16/bf16 case
 * A @ B = (B^T @ A^T)^T
 *
 * original input shape = (1, 1, K)
 * original weight shape = (1, K, N)
 * original output shape = (1, 1, N)
 * step1. Matmul(input, weight, bias) -> Reshape(Matmul(Transpose(weight),
 * Reshape(input), bias))
 *
 *
 * after apply pattern:
 * new weight shape = (1, N, K)
 * new input shape = (1, K, 1)
 * new output shape = (1, N, 1) -reshape-> (1, 1, N)
 */
class MatmulUsePermutePattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  MatmulUsePermutePattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatmulUsePermutePattern",
                                           benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    if (!module::getElementType(input).isa<Float16Type, BFloat16Type>()) {
      return failure();
    }
    if (op.getLeftTranspose() || op.getRightTranspose() ||
        op.getOutputTranspose() || op.getHdimIsBatch()) {
      return failure();
    }
    std::vector<tpu::MatMulOp> sameMatmuls;
    auto inputShape = module::getShape(input);
    if (!isa<top::WeightOp>(op->getOperand(1).getDefiningOp()) ||
        inputShape.size() != 3 || inputShape[0] != 1 || inputShape[1] != 1) {
      return failure();
    }

    // Find all MatMulOps with the same input
    for (auto user : input.getUsers()) {
      if (auto matmulOp = dyn_cast<tpu::MatMulOp>(user)) {
        if (matmulOp.getInput() == input) {
          if (!isa<top::WeightOp>(matmulOp->getOperand(1).getDefiningOp())) {
            continue;
          }
          sameMatmuls.push_back(matmulOp);
        }
      } else {
        return failure();
      }
    }

    if (sameMatmuls.size() == 0) {
      return failure();
    }

    for (auto matmul : sameMatmuls) {
      auto right = matmul.getRight();
      auto right_name = module::getName(right).str();
      auto rightShape = module::getShape(right);
      if (rightShape.size() != 2) {
        return failure();
      }
      auto row = rightShape[0];
      auto col = rightShape[1];
      auto new_right_type =
          RankedTensorType::get({col, row}, module::getElementType(right));
      auto weight_data = right.getDefiningOp<top::WeightOp>().read<uint16_t>();
      // transpose the weight data
      auto trans_weight =
          std::make_shared<std::vector<uint16_t>>(weight_data->size());
      for (int i = 0; i < col; ++i) {
        for (int j = 0; j < row; ++j) {
          (*trans_weight)[i * row + j] = (*weight_data)[j * col + i];
        }
      }

      auto new_right = top::WeightOp::create(matmul, right_name + "_trans",
                                             *trans_weight, new_right_type);
      matmul.setOperand(0, new_right);
    }

    rewriter.setInsertionPointAfter(input.getDefiningOp());
    auto in_reshape_op = rewriter.create<tpu::ReshapeOp>(
        NameLoc::get(
            rewriter.getStringAttr(module::getName(input) + "_reshape")),
        RankedTensorType::get({inputShape[0], inputShape[2], inputShape[1]},
                              module::getElementType(input)),
        ValueRange{input, module::getNoneOp(op)});

    for (auto matmul : sameMatmuls) {
      matmul.setOperand(1, in_reshape_op.getOutput());

      auto resultShape = module::getShape(matmul.getResult());
      // auto oriType = matmul.getResult().getType();
      matmul.getResult().setType(RankedTensorType::get(
          {resultShape[0], resultShape[2], resultShape[1]},
          module::getElementType(matmul.getResult())));

      rewriter.setInsertionPointAfter(matmul);

      auto reshapeType = RankedTensorType::get(
          resultShape, module::getElementType(matmul.getResult()));
      auto matmul_name = module::getName(matmul.getOutput());
      module::setLoc(matmul.getResult(), NameLoc::get(rewriter.getStringAttr(
                                             matmul_name + "_prereshape")));
      auto mm_reshape_op = rewriter.create<tpu::ReshapeOp>(
          NameLoc::get(rewriter.getStringAttr(matmul_name)), reshapeType,
          ValueRange{matmul.getOutput()});

      matmul.getOutput().replaceAllUsesExcept(mm_reshape_op.getOutput(),
                                              {mm_reshape_op});
    }

    return success();
  }
};

/**
 * Use together with MatmulUsePermutePattern
 *
 * Matmul(weight1, input1, bias1) - \                                        /
 * ... Matmul(weight2, input2, bias2) -  | -> concat(B1..Bn) X A -> Slice x n ->
 * - ... Matmul(weight3, input3, bias3) - / \ ...
 *
 */
class MultipleSameActivationMatmulMergePattern
    : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  MultipleSameActivationMatmulMergePattern(mlir::MLIRContext *context,
                                           int benefit)
      : OpRewriterPatternEx<tpu::MatMulOp>(
            context, "MultipleSameActivationMatmulMergePattern", benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto none = module::getNoneOp(op);
    if (op->getUsers().empty()) {
      return failure();
    }

    auto input = op.getOperand(1);
    if (!module::getElementType(input).isa<BFloat16Type, Float16Type>()) {
      return failure();
    }
    if (isa<top::WeightOp>(input.getDefiningOp())) {
      return failure();
    }
    auto inputShape = module::getShape(input);
    if (inputShape.size() != 3 || inputShape[2] != 1 || inputShape[0] != 1) {
      return failure();
    }

    // weight shape may differ
    std::vector<int> weight_rows;
    auto weight_col = inputShape[1];
    std::vector<tpu::MatMulOp> sameMatmuls;
    std::vector<Value> mmWeights;
    std::vector<Value> mmBiases;

    std::vector<std::shared_ptr<std::vector<uint16_t>>> weight_data_set;
    std::vector<std::shared_ptr<std::vector<uint16_t>>> bias_data_set;
    // Find all MatMulOps with the same input
    bool is_first_mm = true;
    Operation *first_mm = none;
    for (auto user : input.getUsers()) {
      if (auto matmulOp = dyn_cast<tpu::MatMulOp>(user)) {
        if (matmulOp.getRight() == input &&
            !module::isNone(matmulOp.getBias()) &&
            isa<top::WeightOp>((matmulOp.getBias()).getDefiningOp())) {
          if (auto cur_weight_op = dyn_cast<top::WeightOp>(
                  (matmulOp.getOperands()[0]).getDefiningOp())) {
            if (is_first_mm) {
              is_first_mm = false;
              first_mm = user;
            } else if (!module::areAttributesEqual(first_mm, user)) {
              return failure();
            }
            weight_rows.push_back(
                module::getShape(cur_weight_op.getResult())[0]);
            weight_data_set.push_back(cur_weight_op.read<uint16_t>());
            auto bias_op =
                cast<top::WeightOp>((matmulOp.getBias()).getDefiningOp());
            std::shared_ptr<std::vector<uint16_t>> bias_data;
            if (module::getElementType(bias_op.getOutput())
                    .isa<Float32Type>()) {
              auto f32_bias = bias_op.read<float>();
              auto count = f32_bias->size();
              bias_data = std::make_shared<std::vector<uint16_t>>(count);
#pragma omp parallel for schedule(static, omp_schedule(count))
              for (uint32_t i = 0; i < count; i++) {
                bias_data->at(i) =
                    module::getElementType(input).isa<Float16Type>()
                        ? f32_to_f16(f32_bias->at(i))
                        : f32_to_bf16(f32_bias->at(i));
              }
            } else {
              bias_data = bias_op.read<uint16_t>();
            }
            bias_data_set.push_back(bias_data);
            sameMatmuls.push_back(matmulOp);
            mmWeights.push_back(matmulOp.getOperand(0));
            mmBiases.push_back(matmulOp.getBias());
          } else {
            return failure();
          }

        } else {
          return failure();
        }
      } else {
        return failure();
      }
    }

    if (sameMatmuls.size() <= 1) {
      return failure();
    }

    rewriter.setInsertionPointAfter(input.getDefiningOp());
    // step1. concat weight, get shape (size*N) x K
    std::string weight_name =
        module::getName(op.getOperands()[0]).str() + "_merged_weight";
    auto new_weight_row =
        std::accumulate(weight_rows.begin(), weight_rows.end(), 0);
    auto weight_type =
        RankedTensorType::get({new_weight_row, weight_col},
                              module::getElementType(op.getOperands()[0]));
    auto weight_size = weight_type.getNumElements();
    auto weight_data = std::make_shared<std::vector<uint16_t>>(weight_size, 0);

    int offset = 0;
    for (int i = 0; i < weight_rows.size(); i++) {
      std::copy_n(weight_data_set[i]->begin(), weight_rows[i] * weight_col,
                  weight_data->begin() + offset);
      offset += weight_rows[i] * weight_col;
    }
    offset = 0;
    auto wret = module::weightFile().addTensor(
        weight_name, (uint16_t *)weight_data->data(), weight_type);
    assert(succeeded(wret));
    auto weight_value = rewriter.create<top::WeightOp>(
        NameLoc::get(rewriter.getStringAttr(weight_name)), weight_type,
        ValueRange{});

    // step2. concat bias, get shape (size*N)
    std::string bias_name =
        module::getName(op.getBias()).str() + "_merged_bias";
    auto bias_type = RankedTensorType::get(
        {new_weight_row}, module::getElementType(op.getOperands()[0]));
    auto bias_size = bias_type.getNumElements();
    auto bias_data = std::make_shared<std::vector<uint16_t>>(bias_size, 0);
    for (int i = 0; i < weight_rows.size(); i++) {
      std::copy_n(bias_data_set[i]->begin(), weight_rows[i],
                  bias_data->begin() + offset);
      offset += weight_rows[i];
    }
    auto bret = module::weightFile().addTensor(
        bias_name, (uint16_t *)bias_data->data(), bias_type);
    assert(succeeded(bret));
    auto bias_value = rewriter.create<top::WeightOp>(
        NameLoc::get(rewriter.getStringAttr(bias_name)), bias_type,
        ValueRange{});
    // step3. create new large MatMulOp, W(N_new x K) @ Ipt(1 x K x 1) +
    // B(N_new) => R(1 x N_new)
    auto newMatmulOp = rewriter.create<tpu::MatMulOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(op.getOutput()).str() + "_merged")),
        RankedTensorType::get({1, new_weight_row, 1},
                              module::getElementType(op.getOutput())),
        ValueRange{weight_value, input, none, none, none}, op->getAttrs());

    auto resultShape = module::getShape(newMatmulOp.getResult());
    rewriter.setInsertionPointAfter(newMatmulOp);
    auto reshapeType =
        RankedTensorType::get({resultShape[0], resultShape[2], resultShape[1]},
                              module::getElementType(newMatmulOp.getResult()));
    auto reshapeOp = rewriter.create<tpu::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(newMatmulOp.getOutput()) + "_new_reshape")),
        reshapeType, ValueRange{newMatmulOp.getOutput()});

    auto add_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(reshapeOp.getOperation()).str() + "_add"));
    auto add_op = rewriter.create<tpu::AddOp>(
        add_loc, reshapeType,
        mlir::ValueRange{reshapeOp.getOutput(), bias_value});
    // step4. slice each original MatMulOp
    std::vector<Operation *> operands;
    auto sliceOffset = 0;
    for (size_t i = 0; i < sameMatmuls.size(); ++i) {
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({0, 1, 2})));
      attrs.push_back(rewriter.getNamedAttr(
          "ends", rewriter.getI64ArrayAttr({1, 1, weight_rows[i]})));
      attrs.push_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr({0, 0, sliceOffset})));
      attrs.push_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr({1, 1, 1})));
      rewriter.setInsertionPointAfter(add_op);

      auto slice_type = RankedTensorType::get(
          {1, 1, weight_rows[i]}, module::getElementType(add_op.getOutput()));
      auto slice_value =
          rewriter
              .create<tpu::SliceOp>(
                  sameMatmuls[i]->getLoc(), slice_type,
                  ValueRange{add_op.getOutput(), none, none, none, none}, attrs)
              .getResult();

      auto reshape_op = dyn_cast<tpu::ReshapeOp>(
          *sameMatmuls[i].getOutput().getUsers().begin());
      reshape_op.getOutput().replaceAllUsesWith(slice_value);
      auto ori_loc = reshape_op.getLoc();
      rewriter.eraseOp(reshape_op);
      slice_value.setLoc(ori_loc);
      sliceOffset += weight_rows[i];
    }

    for (auto op : sameMatmuls) {
      rewriter.eraseOp(op);
    }

    for (auto operand : mmWeights) {
      rewriter.eraseOp(operand.getDefiningOp());
    }

    for (auto operand : mmBiases) {
      rewriter.eraseOp(operand.getDefiningOp());
    }

    return success();
  }
};

// transform group conv to normal conv, when int8/f16/bf16 &&
// input_c<=ic_parallel && isBM1684XFamily()
class GroupConv2NormalConv : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  GroupConv2NormalConv(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "GroupConv2NormalConv",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::Conv2DOp op,
                                    PatternRewriter &rewriter) const override {
    if (!(module::isBM1684XFamily() || module::isBM1690Family()) ||
        !module::isWeight(op.getFilter())) {
      return failure();
    }
    auto data_type = module::getStorageType(op.getFilter());
    if (!(data_type.isBF16() || data_type.isF16() || data_type.isInteger(8))) {
      return failure();
    }
    auto attrs = op.parseParam();
    if (attrs.groups == 1) {
      return failure();
    }
    int ic_parallel = BM168x::ic_num(data_type.getIntOrFloatBitWidth() / 8);
    if (attrs.ic > ic_parallel) {
      return failure();
    }

    if (data_type.isUnsignedInteger(8)) {
      updateFilter<uint8_t>(op, attrs);
    } else if (data_type.isInteger(8)) {
      updateFilter<int8_t>(op, attrs);
    } else {
      updateFilter<uint16_t>(op, attrs);
    }
    op.setGroup(1);
    return success();
  }

private:
  template <typename T>
  void updateFilter(tpu::Conv2DOp op, const conv_attr_t &p) const {
    int gic = p.ic / p.groups;
    int goc = p.oc / p.groups;
    int old_ic_num = gic * p.kh * p.kw;
    int new_ic_num = p.ic * p.kh * p.kw;
    auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
    auto filter_data = filterOp.read<T>();
    auto filter_size = filter_data->size();
    auto new_data = std::make_shared<std::vector<T>>(filter_size * p.groups,
                                                     op.getKernelZp());
    for (int i = 0; i < p.oc; i++) {
      auto begin = filter_data->begin() + old_ic_num * i;
      auto end = begin + old_ic_num;
      int group_idx = i / goc;
      auto to = new_data->begin() + new_ic_num * i + old_ic_num * group_idx;
      std::copy(begin, end, to);
    }
    auto new_type =
        module::getTypeLike(op.getFilter(), {p.oc, p.ic, p.kh, p.kw});
    auto new_filter =
        top::WeightOp::create(op, "filter_g2normal", *new_data, new_type);
    op->setOperand(1, new_filter);
  }
};

// reorder op when transpose is before mulconst/cast/softmax to optimize bert
// TODO: may be merged into PermuteReorderPattern
class PermuteAddWeightReorderPattern
    : public OpRewriterPatternEx<tpu::PermuteOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteAddWeightReorderPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(
            context, "PermuteAddWeightReorderPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    //  if (module::isAsymmetric()) {
    //    return failure();
    //  }

    if (op->hasOneUse() == false) {
      return failure();
    }
    std::vector<int64_t> ps = {0, 2, 1, 3};
    auto order = module::getI64Array(op.getOrder());
    if (*order != ps) {
      return failure();
    }

    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    auto nextOp = *op.getOutput().user_begin();
    if (nextOp->hasOneUse() == false) {
      return failure();
    }
    if (auto add_op = dyn_cast<tpu::AddOp>(nextOp)) {
      /**
       * weight        ->         permuted_weight   ->
       *               -> Add =>                    -> Add -> perm
       * input -> perm ->         input             ->
       *
       */
      auto inB = add_op.getInputs()[1];
      if (!module::isWeight(inB)) {
        return failure();
      }
      std::vector<int64_t> inB_shape = module::getShape(inB);
      std::vector<int64_t> new_inB_shape = {inB_shape[0], inB_shape[2],
                                            inB_shape[1], inB_shape[3]};
      auto newType = module::getTypeLike(inB, new_inB_shape);
      auto weight_op = inB.getDefiningOp<top::WeightOp>();
      auto weight_type = module::getElementType(weight_op.getOutput());
      if (weight_type.isF16() || weight_type.isBF16()) {
        auto weight_data = weight_op.read<uint16_t>();
        auto weight_tp =
            std::make_shared<std::vector<uint16_t>>(weight_data->size(), 0);
        function_permute(weight_data->data(), weight_tp->data(), inB_shape, ps);
        auto weight = tpu_mlir::top::WeightOp::create<uint16_t>(
            add_op, "transposed_add_weight", *weight_tp, newType);
        add_op.setOperand(1, weight);
      } else if (weight_type.isF32()) {
        auto weight_data = weight_op.read<float>();
        auto weight_tp =
            std::make_shared<std::vector<float>>(weight_data->size(), 0);
        function_permute(weight_data->data(), weight_tp->data(), inB_shape, ps);
        auto weight = tpu_mlir::top::WeightOp::create<float>(
            add_op, "transposed_add_weight", *weight_tp, newType);
        add_op.setOperand(1, weight);
      } else if (weight_type.isInteger(8)) {
        auto weight_data = weight_op.read<uint8_t>();
        auto weight_tp =
            std::make_shared<std::vector<uint8_t>>(weight_data->size(), 0);
        function_permute(weight_data->data(), weight_tp->data(), inB_shape, ps);
        auto weight = tpu_mlir::top::WeightOp::create<uint8_t>(
            add_op, "transposed_add_weight", *weight_tp, newType);
        add_op.setOperand(1, weight);
      }

      newType = RankedTensorType::get(
          in_shape, module::getElementType(add_op.getOutput()));
      add_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(add_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(add_op.getOutput()));
      auto out_loc = add_op.getLoc(); // keep out location unchanged.
      module::setLocSuffix(add_op, "_trans");
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{add_op.getOutput(), module::getNoneOp(add_op)}, attrs);
      rewriter.replaceAllUsesExcept(add_op.getOutput(), new_op.getOutput(),
                                    {new_op});
      rewriter.eraseOp(op);
      return success();
    } else if (auto mul_op = dyn_cast<tpu::MulOp>(nextOp)) {
      auto inB = mul_op.getInputs()[1];
      if (!module::isWeight(inB)) {
        return failure();
      }
      auto inB_shape = module::getShape(inB);
      if (inB_shape.size() < 2 || inB_shape[1] != 1) {
        return failure();
      }
      std::vector<int64_t> new_inB_shape = {inB_shape[0], inB_shape[2],
                                            inB_shape[1], inB_shape[3]};
      module::setShape(inB, new_inB_shape);
      Value mul_out = mul_op.getOutput();
      module::setShape(mul_out, in_shape);

      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(mul_op);
      auto newType = module::getTypeLike(mul_out, out_shape);
      auto out_loc = mul_op.getLoc(); // keep out location unchanged.
      module::setLocSuffix(mul_op, "trans");
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType, ValueRange{mul_out, module::getNoneOp(mul_op)},
          attrs);
      rewriter.replaceAllUsesExcept(mul_out, new_op.getOutput(), {new_op});
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

class PermuteRopeWeightReorderPattern
    : public OpRewriterPatternEx<tpu::PermuteOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteRopeWeightReorderPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(
            context, "PermuteRopeWeightReorderPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    //  if (module::isAsymmetric()) {
    //    return failure();
    //  }

    if (op->hasOneUse() == false) {
      return failure();
    }
    std::vector<int64_t> ps = {0, 2, 1, 3};
    auto order = module::getI64Array(op.getOrder());
    if (*order != ps) {
      return failure();
    }

    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    auto nextOp = *op.getOutput().user_begin();
    if (nextOp->hasOneUse() == false) {
      return failure();
    }
    if (auto rope_op = dyn_cast<tpu::RopeOp>(nextOp)) {
      /**
       * weight        ->         permuted_weight   ->
       *               -> Rope =>                    -> Rope -> perm
       * input -> perm ->         input             ->
       *
       */
      // #############################  inW0   ###############################
      auto inW0 = rope_op.getInput2();
      if (!module::isWeight(inW0)) {
        return failure();
      }
      std::vector<int64_t> inW0_shape = module::getShape(inW0);
      std::vector<int64_t> new_inW0_shape = {inW0_shape[0], inW0_shape[2],
                                             inW0_shape[1], inW0_shape[3]};
      auto newType = module::getTypeLike(inW0, new_inW0_shape);
      auto weight0_op = inW0.getDefiningOp<top::WeightOp>();
      auto weight0_type = module::getElementType(weight0_op.getOutput());
      if (weight0_type.isF16() || weight0_type.isBF16()) {
        auto weight0_data = weight0_op.read<uint16_t>();
        auto weight0_tp =
            std::make_shared<std::vector<uint16_t>>(weight0_data->size(), 0);
        function_permute(weight0_data->data(), weight0_tp->data(), inW0_shape,
                         ps);
        auto weight0 = tpu_mlir::top::WeightOp::create<uint16_t>(
            rope_op, "transposed_rope_weight0", *weight0_tp, newType);
        rope_op.setOperand(1, weight0);
      } else if (weight0_type.isF32()) {
        auto weight0_data = weight0_op.read<float>();
        auto weight0_tp =
            std::make_shared<std::vector<float>>(weight0_data->size(), 0);
        function_permute(weight0_data->data(), weight0_tp->data(), inW0_shape,
                         ps);
        auto weight0 = tpu_mlir::top::WeightOp::create<float>(
            rope_op, "transposed_rope_weight0", *weight0_tp, newType);
        rope_op.setOperand(1, weight0);
      } else if (weight0_type.isInteger(8)) {
        auto weight0_data = weight0_op.read<uint8_t>();
        auto weight0_tp =
            std::make_shared<std::vector<uint8_t>>(weight0_data->size(), 0);
        function_permute(weight0_data->data(), weight0_tp->data(), inW0_shape,
                         ps);
        auto weight0 = tpu_mlir::top::WeightOp::create<uint8_t>(
            rope_op, "transposed_rope_weight0", *weight0_tp, newType);
        rope_op.setOperand(1, weight0);
      }

      // #############################  inW1   ###############################
      auto inW1 = rope_op.getInput3();
      if (!module::isWeight(inW1)) {
        return failure();
      }
      std::vector<int64_t> inW1_shape = module::getShape(inW1);
      std::vector<int64_t> new_inW1_shape = {inW1_shape[0], inW1_shape[2],
                                             inW1_shape[1], inW1_shape[3]};
      auto newType1 = module::getTypeLike(inW1, new_inW1_shape);
      auto weight1_op = inW1.getDefiningOp<top::WeightOp>();
      auto weight1_type = module::getElementType(weight1_op.getOutput());
      if (weight1_type.isF16() || weight1_type.isBF16()) {
        auto weight1_data = weight1_op.read<uint16_t>();
        auto weight1_tp =
            std::make_shared<std::vector<uint16_t>>(weight1_data->size(), 0);
        function_permute(weight1_data->data(), weight1_tp->data(), inW1_shape,
                         ps);
        auto weight1 = tpu_mlir::top::WeightOp::create<uint16_t>(
            rope_op, "transposed_rope_weight1", *weight1_tp, newType1);
        rope_op.setOperand(2, weight1);
      } else if (weight1_type.isF32()) {
        auto weight1_data = weight1_op.read<float>();
        auto weight1_tp =
            std::make_shared<std::vector<float>>(weight1_data->size(), 0);
        function_permute(weight1_data->data(), weight1_tp->data(), inW1_shape,
                         ps);
        auto weight1 = tpu_mlir::top::WeightOp::create<float>(
            rope_op, "transposed_rope_weight1", *weight1_tp, newType1);
        rope_op.setOperand(2, weight1);
      } else if (weight1_type.isInteger(8)) {
        auto weight1_data = weight1_op.read<uint8_t>();
        auto weight1_tp =
            std::make_shared<std::vector<uint8_t>>(weight1_data->size(), 0);
        function_permute(weight1_data->data(), weight1_tp->data(), inW1_shape,
                         ps);
        auto weight1 = tpu_mlir::top::WeightOp::create<uint8_t>(
            rope_op, "transposed_rope_weight1", *weight1_tp, newType1);
        rope_op.setOperand(2, weight1);
      }

      newType = RankedTensorType::get(
          in_shape, module::getElementType(rope_op.getOutput()));
      rope_op.getOutput().setType(newType);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(rope_op);
      newType = RankedTensorType::get(
          out_shape, module::getElementType(rope_op.getOutput()));
      auto out_loc = rope_op.getLoc(); // keep out location unchanged.
      module::setLocSuffix(rope_op, "_trans");
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{rope_op.getOutput(), module::getNoneOp(rope_op)}, attrs);
      rewriter.replaceAllUsesExcept(rope_op.getOutput(), new_op.getOutput(),
                                    {new_op});
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

// reorder op when transpose is before mulconst
// permute order = {0,2,3,1}
class PermuteMulconstSwap : public OpRewriterPatternEx<tpu::PermuteOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteMulconstSwap(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteMulconstSwap",
                                            benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {

    if (op->hasOneUse() == false) {
      return failure();
    }
    std::vector<int64_t> ps = {0, 2, 3, 1};
    auto order = module::getI64Array(op.getOrder());
    if (*order != ps) {
      return failure();
    }

    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    auto nextOp = *op.getOutput().user_begin();
    if (nextOp->hasOneUse() == false) {
      return failure();
    }

    auto next_nextOp = *nextOp->getResult(0).getUsers().begin();
    auto matmul_op = dyn_cast_or_null<tpu::MatMulOp>(next_nextOp);
    if (matmul_op) {
      auto matmul_op = dyn_cast<tpu::MatMulOp>(next_nextOp);
      auto left = matmul_op.getInput();
      if (!isa<tpu::PermuteOp>(left.getDefiningOp())) {
        return failure();
      }
    } else {
      return failure();
    }

    if (isa<tpu::MulShiftOp, tpu::MulConstOp>(nextOp)) {
      auto mulconst_or_mulshift_op = nextOp;
      Value mulconst_or_mulshift_out = nextOp->getOpResult(0);
      module::setShape(mulconst_or_mulshift_out, in_shape);
      op.replaceAllUsesWith(op.getInput());
      rewriter.setInsertionPointAfter(mulconst_or_mulshift_op);
      auto newType = module::getTypeLike(mulconst_or_mulshift_out, out_shape);
      auto out_loc =
          mulconst_or_mulshift_op->getLoc(); // keep out location unchanged.
      module::setLocSuffix(mulconst_or_mulshift_op, "trans");
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
      auto new_op = rewriter.create<tpu::PermuteOp>(
          out_loc, newType,
          ValueRange{mulconst_or_mulshift_out,
                     module::getNoneOp(mulconst_or_mulshift_op)},
          attrs);
      rewriter.replaceAllUsesExcept(mulconst_or_mulshift_out,
                                    new_op.getOutput(), {new_op});
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

/**
 * input0 + Permute \              => input0           \
 *                   => MaskedFill =>                   => MaskedFill + Permute
 * input1           /              => input1 + Permute /
 */
class MaskedFillPermuteMove : public OpRewriterPatternEx<tpu::MaskedFillOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  MaskedFillPermuteMove(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MaskedFillOp>(context, "MaskedFillPermuteMove",
                                               benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::MaskedFillOp op,
                                    PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getBrn());
    auto condition_shape = module::getShape(op.getCond());
    if (input_shape != condition_shape) {
      return failure();
    }
    auto op_name = module::getName(op.getOutput()).str();
    if (op_name.find("_masked_fill") != std::string::npos) {
      return failure();
    }
    auto none_op = module::getNoneOp(op);
    std::vector<bool> is_permute;
    assert(op->getNumOperands() == 2);
    tpu::PermuteOp permute_op;
    for (auto opd : op->getOperands()) {
      Operation *op_ = opd.getDefiningOp();
      if (isa<tpu::PermuteOp>(op_)) {
        is_permute.push_back(true);
        permute_op = dyn_cast<tpu::PermuteOp>(op_);
      } else {
        is_permute.push_back(false);
      }
    }
    if (is_permute[0] == is_permute[1]) {
      return failure();
    }
    auto permute_attr = permute_op->getAttrs();
    auto permute_order = *module::getI64Array(permute_op.getOrder());
    std::vector<int64_t> inv_order(permute_order.size());
    for (int i = 0; i < permute_order.size(); ++i) {
      inv_order[permute_order[i]] = i;
    }
    int need_permute = is_permute[0] ? 1 : 0;
    auto need_permute_op = op->getOperand(need_permute);

    auto type = permute_op.getInput().getType();
    auto name = module::getName(need_permute_op);
    std::vector<NamedAttribute> attrs;

    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(inv_order)));

    int user_count = 0;
    for (auto j : need_permute_op.getUsers()) {
      if (isa<tpu::PermuteOp>(j)) {
        user_count++;
      }
    }
    auto loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_permute" +
                                                   std::to_string(user_count)));
    auto new_permute_op = rewriter.create<tpu::PermuteOp>(
        loc, type, ValueRange{need_permute_op, none_op}, attrs);
    auto masked_fill_attrs = op->getAttrs();
    loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(need_permute_op).str() +
                               "_masked_fill" + std::to_string(user_count)));
    Value cond, brn;
    if (is_permute[0]) {
      cond = permute_op.getInput();
      brn = new_permute_op.getOutput();
    } else {
      cond = new_permute_op.getOutput();
      brn = permute_op.getInput();
    }
    rewriter.setInsertionPointAfterValue(new_permute_op.getOutput());
    auto new_masked_fill_op = rewriter.create<tpu::MaskedFillOp>(
        loc, type, ValueRange{cond, brn}, masked_fill_attrs);
    rewriter.replaceAllUsesWith(permute_op, new_masked_fill_op.getOutput());
    rewriter.eraseOp(permute_op);
    rewriter.setInsertionPointAfterValue(new_masked_fill_op.getOutput());
    auto post_permute_op = rewriter.create<tpu::PermuteOp>(
        op.getLoc(), op.getOutput().getType(),
        ValueRange{new_masked_fill_op.getOutput(),
                   module::getNoneOp(new_masked_fill_op)},
        permute_attr);
    rewriter.replaceAllUsesWith(op.getOutput(), post_permute_op.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * Optimize for permute fuse in sam-vit-base encoder
 *
 * permute -> (reshape) -> \
 *                          Add => Add -> permute
 * permute -> (reshape) -> /
 */
class MovePermuteAfterAdd : public OpRewriterPatternEx<tpu::AddOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  MovePermuteAfterAdd(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::AddOp>(context, "MovePermuteAfterAdd",
                                        benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::AddOp op,
                                    PatternRewriter &rewriter) const override {
    auto l_op = op.getOperand(0).getDefiningOp();
    auto r_op = op.getOperand(1).getDefiningOp();

    if (isa<tpu::ReshapeOp>(l_op) && isa<tpu::ReshapeOp>(r_op)) {
      auto l_reshape_op = cast<tpu::ReshapeOp>(l_op);
      auto r_reshape_op = cast<tpu::ReshapeOp>(r_op);
      if (!isa<tpu::PermuteOp>(l_reshape_op.getInput().getDefiningOp()) ||
          !isa<tpu::PermuteOp>(r_reshape_op.getInput().getDefiningOp())) {
        return failure();
      }
      if (!MoveReshapeAfterAdd(l_reshape_op, r_reshape_op, op, rewriter)) {
        return failure();
      }
      l_op = op.getOperand(0).getDefiningOp();
      r_op = op.getOperand(1).getDefiningOp();
    }

    auto l_permute_op = dyn_cast<tpu::PermuteOp>(l_op);
    auto r_permute_op = dyn_cast<tpu::PermuteOp>(r_op);
    if (!l_permute_op || !r_permute_op)
      return failure();
    auto l_in_shape = module::getShape(l_permute_op.getInput()).vec();
    auto r_in_shape = module::getShape(r_permute_op.getInput()).vec();
    if (l_in_shape.size() != r_in_shape.size())
      return failure();
    auto l_permute_order = *module::getI64Array(l_permute_op.getOrder());
    auto r_permute_order = *module::getI64Array(r_permute_op.getOrder());
    if (l_permute_order != r_permute_order)
      return failure();
    auto loc = op.getLoc();
    op.setOperand(0, l_permute_op.getInput());
    op.setOperand(1, r_permute_op.getInput());
    auto output = op.getOutput();
    auto output_type = output.getType();
    std::vector<int64_t> new_shape;
    for (int i = 0; i < l_in_shape.size(); ++i) {
      new_shape.push_back(std::max(l_in_shape[i], r_in_shape[i]));
    }
    module::setShape(output, new_shape);
    module::setLocSuffix(op, "before_permute");

    if (l_permute_op.getOutput().getUsers().empty()) {
      rewriter.eraseOp(l_permute_op);
    }
    if (r_permute_op.getOutput().getUsers().empty()) {
      rewriter.eraseOp(r_permute_op);
    }

    rewriter.setInsertionPointAfterValue(output);
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr(
        "order", rewriter.getI64ArrayAttr(l_permute_order)));
    auto new_permute_op = rewriter.create<tpu::PermuteOp>(
        loc, output_type, ValueRange{output, module::getNoneOp(op)}, attrs);
    rewriter.replaceAllUsesExcept(output, new_permute_op.getOutput(),
                                  new_permute_op);
    return success();
  }

private:
  bool MoveReshapeAfterAdd(tpu::ReshapeOp &l_reshape_op,
                           tpu::ReshapeOp &r_reshape_op, tpu::AddOp &add_op,
                           PatternRewriter &rewriter) const {
    if (l_reshape_op.getOutput().hasOneUse() == false ||
        r_reshape_op.getOutput().hasOneUse() == false) {
      return false;
    }
    auto l_in_shape = module::getShape(l_reshape_op.getInput()).vec();
    auto r_in_shape = module::getShape(r_reshape_op.getInput()).vec();
    if (l_in_shape != r_in_shape)
      return false;
    auto l_out_shape = module::getShape(l_reshape_op.getOutput()).vec();
    auto r_out_shape = module::getShape(r_reshape_op.getOutput()).vec();
    if (l_out_shape != r_out_shape)
      return false;
    auto loc = add_op.getLoc();
    add_op.setOperand(0, l_reshape_op.getInput());
    add_op.setOperand(1, r_reshape_op.getInput());
    auto output = add_op.getOutput();
    module::setShape(output, l_in_shape);
    module::setLocSuffix(add_op, "before_reshape");

    rewriter.setInsertionPointAfterValue(output);
    auto reshape_type = module::getTypeLike(output, l_out_shape);
    auto new_reshape_op =
        rewriter.create<tpu::ReshapeOp>(loc, reshape_type, ValueRange{output});
    rewriter.replaceAllUsesExcept(output, new_reshape_op.getOutput(),
                                  new_reshape_op);
    rewriter.eraseOp(l_reshape_op);
    rewriter.eraseOp(r_reshape_op);
    return true;
  }
};

// reorder op when reshapeOp is before matmul/mulconst/cast/softmax op to
// eliminate reshapeOp
// copied from lib/Dialect/Top/Transforms/ProcessorOptimize/OptimizeBM1684X.cpp
class TpuReshapeReorderPattern : public OpRewriterPatternEx<tpu::ReshapeOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  TpuReshapeReorderPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ReshapeOp>(context, "TpuReshapeReorderPattern",
                                            benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto next_op_ = *output.user_begin();
    auto ishape = module::getShape(op.getInput());
    auto oshape = module::getShape(op.getOutput());
    if (ishape.size() == 4 && ishape[0] == 1 && ishape[1] == 1 &&
        oshape.size() == 3 && ishape[2] == oshape[1] &&
        ishape[3] == oshape[2]) {
      // InsertReshape optimize, do not shape reorder
      return failure();
    }

    if (auto next_op = dyn_cast<tpu::MatMulOp>(next_op_)) {
      // right is from Reshape too
      auto left = next_op.getInput();
      auto right = next_op.getRight();
      auto right_op_ = right.getDefiningOp();
      auto right_op = dyn_cast<tpu::ReshapeOp>(right_op_);
      if (op != left.getDefiningOp() || !right_op) {
        return failure();
      }
      // check left and right are both Reshape(n, c, h, w) --> (nxc, h, w)
      auto lshape_ = SmallVector<int64_t>(module::getShape(op.getInput()));
      auto lshape = module::getShape(left);
      if (!(lshape.size() == 3 && lshape_.size() == 4 &&
            lshape[0] == lshape_[0] * lshape_[1] && lshape[1] == lshape_[2] &&
            lshape[2] == lshape_[3])) {
        return failure();
      }
      auto rshape_ = module::getShape(right_op.getInput());
      auto rshape = SmallVector<int64_t>(module::getShape(right));
      if (!(rshape.size() == 3 && rshape_.size() == 4 &&
            rshape[0] == rshape_[0] * rshape_[1] && rshape[1] == rshape_[2] &&
            rshape[2] == rshape_[3])) {
        return failure();
      }
      if (lshape_[0] != rshape_[0] || lshape_[1] != rshape_[1]) {
        return failure();
      }

      // remove left and right ReshapeOp
      op.replaceAllUsesWith(op.getInput());
      right_op.replaceAllUsesWith(right_op.getInput());

      // Update MatMul output shape
      // and update loc to avoid comparing
      auto next_out = next_op.getOutput();
      auto ori_out_type = next_out.getType();
      auto oshape = module::getShape(next_out);
      std::vector<int64_t> new_oshape{lshape_[0], lshape_[1], oshape[1],
                                      oshape[2]};
      module::setShape(next_out, new_oshape);
      auto ori_loc = next_op.getLoc();
      module::setLocSuffix(next_op, "Reshape");

      // Add ReshapeOp after MatMul
      rewriter.setInsertionPointAfterValue(next_out);
      auto new_reshape_op = rewriter.create<tpu::ReshapeOp>(
          ori_loc, ori_out_type, ValueRange{next_out});
      rewriter.replaceAllUsesExcept(next_out, new_reshape_op.getOutput(),
                                    new_reshape_op);
      rewriter.eraseOp(op);
      rewriter.eraseOp(right_op);
      return success();
    } else if (isa<tpu::MulConstOp, tpu::CastOp, tpu::SoftmaxOp>(next_op_)) {
      // check input is Reshape(n, c, h, w) --> (nxc, h, w)
      auto next_ishape = module::getShape(op.getOutput());
      if (!(next_ishape.size() == 3 && ishape.size() == 4 &&
            next_ishape[0] == ishape[0] * ishape[1] &&
            next_ishape[1] == ishape[2] && next_ishape[2] == ishape[3])) {
        return failure();
      }
      // check next_op param
      if (auto next_op = dyn_cast<tpu::SoftmaxOp>(next_op_)) {
        int64_t axis = next_op.getAxis();
        if (axis != 2 || axis == -1) {
          return failure();
        }
      }

      // remove ReshapeOp
      op.replaceAllUsesWith(op.getInput());

      // update next_op output shape and modify loc name to avoid comparing
      auto next_out = next_op_->getResult(0);
      auto ori_out_type = next_out.getType();
      auto ori_loc = next_op_->getLoc();
      module::setShape(next_out, ishape);
      module::setLocSuffix(next_op_, "Reshape");

      // Add ReshapeOp after MulConst/Cast/Softmax
      rewriter.setInsertionPointAfterValue(next_out);
      auto new_reshape_op = rewriter.create<tpu::ReshapeOp>(
          ori_loc, ori_out_type, ValueRange{next_out});
      rewriter.replaceAllUsesExcept(next_out, new_reshape_op.getOutput(),
                                    new_reshape_op);

      if (auto next_op = dyn_cast<tpu::SoftmaxOp>(next_op_)) {
        next_op->setAttr("axis", rewriter.getSI32IntegerAttr(3));
      }
      rewriter.eraseOp(op);
      return success();
    } else if (auto next_op = dyn_cast<tpu::ReshapeOp>(next_op_)) {
      auto next_oshape = module::getShape(next_op.getOutput());
      if (ishape != next_oshape) {
        return failure();
      }

      op.replaceAllUsesWith(op.getInput());
      next_op.replaceAllUsesWith(next_op.getInput());
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

// permute + permute or permute + reshape + permute
// copied from lib/Dialect/Top/Canonicalize/Permute.cpp (e41cc7c5)
struct PermuteFuse : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteFuse(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteFuse", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    auto rop = dyn_cast<tpu::ReshapeOp>(in.getDefiningOp());
    if (rop) {
      in = rop.getInput();
      if (in.hasOneUse() == false) {
        return failure();
      }
    }
    auto permute_op = dyn_cast<tpu::PermuteOp>(in.getDefiningOp());
    if (!permute_op) {
      return failure();
    }
    // op order
    std::vector<int64_t> in0_shape = module::getShape(permute_op.getInput());
    auto in0_order = module::getI64Array(permute_op.getOrder());
    std::vector<int64_t> in1_shape = module::getShape(op.getInput());
    auto in1_order = module::getI64Array(op.getOrder());
    std::vector<int64_t> out1_shape = module::getShape(op.getOutput());
    std::vector<int64_t> in0_shape_fix;
    std::vector<int64_t> in0_order_fix;
    std::vector<int64_t> out0_shape_fix;
    std::vector<int64_t> in1_shape_fix;
    std::vector<int64_t> in1_order_fix;
    int to_dim;
    for (to_dim = 2; to_dim <= 5; to_dim++) {
      auto ret = permute_reset(in0_shape, *in0_order, in0_shape_fix,
                               in0_order_fix, to_dim);
      if (ret == false) {
        continue;
      }
      ret = permute_reset(in1_shape, *in1_order, in1_shape_fix, in1_order_fix,
                          to_dim);
      if (ret == false) {
        continue;
      }
      break;
    }
    if (to_dim > 5) {
      return failure();
    }
    for (auto o : in0_order_fix) {
      out0_shape_fix.push_back(in0_shape_fix[o]);
    }
    if (in1_shape_fix != out0_shape_fix) {
      return failure();
    }
    // test
    std::vector<int64_t> origin_data;
    for (int64_t i = 0; i < to_dim; i++) {
      origin_data.push_back(i);
    }
    std::vector<int64_t> result0_data;
    for (auto o : in0_order_fix) {
      result0_data.push_back(origin_data[o]);
    }
    std::vector<int64_t> result1_data;
    for (auto o : in1_order_fix) {
      result1_data.push_back(result0_data[o]);
    }
    if (result1_data != origin_data) {
      return failure();
    }
    // bingo !
    if (out1_shape == in0_shape) {
      op.getOutput().replaceAllUsesWith(permute_op.getInput());
      rewriter.eraseOp(op);
      rewriter.eraseOp(permute_op);
    } else {
      auto loc = module::getLocLike(permute_op.getInput(), "Reshape");
      rewriter.setInsertionPoint(op);
      auto rs_op = rewriter.create<tpu::ReshapeOp>(
          loc, op.getOutput().getType(), ValueRange{permute_op.getInput()});
      op.getOutput().replaceAllUsesWith(rs_op.getOutput());
      rewriter.eraseOp(op);
    }
    if (rop) {
      rewriter.eraseOp(rop);
    }
    return success();
  }
};

// permute1 + permute2
// permute_order[0,2,1,3]!=  permute2_order[0,1,3,2]
struct PermuteFuse2 : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteFuse2(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteFuse2", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    auto permute_op = dyn_cast<tpu::PermuteOp>(in.getDefiningOp());
    if (!permute_op) {
      return failure();
    }
    // op order
    auto in0_order = module::getI64Array(permute_op.getOrder());
    auto in1_order = module::getI64Array(op.getOrder());
    if (in0_order == in1_order || in0_order->size() != in1_order->size()) {
      return failure();
    }
    // strict restrictions
    if (in1_order->size() == 4) {
      if (false == (in1_order->at(0) == 0 && in1_order->at(1) == 1 &&
                    in1_order->at(2) == 3 && in1_order->at(3) == 2) ||
          false == (in0_order->at(0) == 0 && in0_order->at(1) == 2 &&
                    in0_order->at(2) == 1 && in0_order->at(3) == 3)) {
        return failure();
      }
    }

    std::vector<int64_t> new_order;
    for (auto o : *in1_order) {
      new_order.push_back((*in0_order)[o]);
    }
    permute_op.getOutput().replaceAllUsesWith(permute_op.getInput());
    op->setAttr("order", rewriter.getI64ArrayAttr(new_order));
    rewriter.eraseOp(permute_op);
    return success();
  }
};

// Calculate `indices_coeff` for GatherElementsOp when axis != indices_dims - 1
//               / 1, i = axis
// axis_flag[i] =
//               \ 0, else
// input_stride[i] = input_shape[i-1] * ... * input_shape[0]
// indices_coeff[i0][i1]...[in-1] = i0 * input_stride[0] * axis_flag[i] + ... +
// in-1 * input_stride[n-1] * axis_flag[n-1]
struct GatherElementsPattern
    : public OpRewriterPatternEx<tpu::GatherElementsOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  GatherElementsPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::GatherElementsOp>(
            context, "GatherElementsPattern", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::GatherElementsOp op,
                                    PatternRewriter &rewriter) const override {

    auto indices = op.getIndices();
    auto indices_shape = module::getShape(indices);
    auto indices_dims = indices_shape.size();
    auto axis = op.getAxis();
    if (axis == indices_dims - 1) {
      return failure();
    }
    if (!op.getIndicesCoeff().getType().isa<NoneType>()) {
      return failure();
    }
    auto input = op.getInput();
    auto input_shape = module::getShape(input);
    std::vector<Value> operands;
    operands.push_back(op.getInput());
    operands.push_back(indices);

    auto indice_type = module::getElementType(indices);
    auto type = RankedTensorType::get(indices_shape, indice_type);
    int indices_shape8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int input_shape8[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    for (int i = 0; i < indices_dims; ++i) {
      indices_shape8[i] = indices_shape[i];
      input_shape8[i] = input_shape[i];
    }
    std::vector<int> indices_coeff;

    int tmp = 0;
    // loop for 8 times
    for (int i0 = 0; i0 < indices_shape8[0]; ++i0) {
      int tmp0 = 0;
      tmp0 += axis == 0 ? 0 : i0;
      tmp0 *= input_shape8[1];
      for (int i1 = 0; i1 < indices_shape8[1]; ++i1) {
        int tmp1 = tmp0;
        tmp1 += axis == 1 ? 0 : i1;
        tmp1 *= input_shape8[2];
        for (int i2 = 0; i2 < indices_shape8[2]; ++i2) {
          int tmp2 = tmp1;
          tmp2 += axis == 2 ? 0 : i2;
          tmp2 *= input_shape8[3];
          for (int i3 = 0; i3 < indices_shape8[3]; ++i3) {
            int tmp3 = tmp2;
            tmp3 += axis == 3 ? 0 : i3;
            tmp3 *= input_shape8[4];
            for (int i4 = 0; i4 < indices_shape8[4]; ++i4) {
              int tmp4 = tmp3;
              tmp4 += axis == 4 ? 0 : i4;
              tmp4 *= input_shape8[5];
              for (int i5 = 0; i5 < indices_shape8[5]; ++i5) {
                int tmp5 = tmp4;
                tmp5 += axis == 5 ? 0 : i5;
                tmp5 *= input_shape8[6];
                for (int i6 = 0; i6 < indices_shape8[6]; ++i6) {
                  int tmp6 = tmp5;
                  tmp6 += axis == 6 ? 0 : i6;
                  tmp6 *= input_shape8[7];
                  for (int i7 = 0; i7 < indices_shape8[7]; ++i7) {
                    tmp++;
                    int tmp7 = tmp6;
                    tmp7 += i7;
                    indices_coeff.push_back(tmp7);
                    // llvm::outs() << tmp << " " << tmp7 << "\n";
                  }
                }
              }
            }
          }
        }
      }
    }

    auto indices_coeff_op =
        top::WeightOp::create(op, "indices_coeff", indices_coeff, type);
    operands.push_back(indices_coeff_op);

    operands.push_back(op.getBuffer());
    rewriter.setInsertionPointAfter(op);
    auto new_op = rewriter.create<tpu::GatherElementsOp>(
        op.getLoc(), op.getResult().getType(), operands, op->getAttrs());
    op.getOutput().replaceAllUsesWith(new_op.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Try insert tile since shapes cannot merge to 4d in some case
 */
template <typename TyOp>
struct TryInsertTileBinaryPattern : public OpRewriterPatternEx<TyOp> {
  // using OpRewriterPatternEx<TyOp>::OpRewriterPatternEx;
  TryInsertTileBinaryPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<TyOp>(context, "TryInsertTileBinaryPattern",
                                  benifit) {}

  bool can_be_merged(int64_t a1, int64_t a2, int64_t b1, int64_t b2) const {
    // case 0: both dims are same --- always true
    if (a1 == b1 && a2 == b2)
      return true;
    // case 1: only one dim is same --- only when another is 1 can be merged
    if ((a1 == b1 && a2 != b2 && a1 == 1) || (a1 != b1 && a2 == b2 && a2 == 1))
      return true;
    // case 2: both dims are not same --- only a or b broadcast can be merged
    if (a1 != b1 && a2 != b2 && (a1 == a2 || b1 == b2))
      return true;
    return false;
  }

  static inline void merge_two_dims(std::vector<int64_t> &ashape,
                                    std::vector<int64_t> &bshape, int dims,
                                    int d_th) {
    ashape[d_th] *= ashape[d_th + 1];
    bshape[d_th] *= bshape[d_th + 1];
    for (int i = d_th + 1; i < dims - 1; i++) {
      ashape[i] = ashape[i + 1];
      bshape[i] = bshape[i + 1];
    }
  }

  bool canMergeTo4D(const std::vector<int64_t> &ashape,
                    const std::vector<int64_t> &bshape, int shape_dim) const {
    std::vector<int64_t> ashape_(8, 1);
    std::vector<int64_t> bshape_(8, 1);
    for (int i = 0; i < ashape.size(); i++) {
      ashape_[i] = ashape[i];
    }
    for (int i = 0; i < bshape.size(); i++) {
      bshape_[i] = bshape[i];
    }
    if (shape_dim > 4) {
      int i = 0;
      while (i < shape_dim - 1) {
        if (can_be_merged(ashape_[i], ashape_[i + 1], bshape_[i],
                          bshape_[i + 1])) {
          merge_two_dims(ashape_, bshape_, shape_dim, i);
          --shape_dim;
        } else {
          ++i;
        }
        if (shape_dim == 4)
          break;
      }
    }
    return shape_dim <= 4;
  }

  bool needBroadcast(const std::vector<int64_t> &shape1,
                     const std::vector<int64_t> &shape2) const {
    int dim1 = shape1.size();
    int dim2 = shape2.size();
    int maxDim = std::max(dim1, dim2);
    for (int i = 1; i <= maxDim; ++i) {
      int size1 = (dim1 - i >= 0) ? shape1[dim1 - i] : 1;
      int size2 = (dim2 - i >= 0) ? shape2[dim2 - i] : 1;
      if (size1 != size2 && (size1 != 1 || size2 != 1)) {
        return true;
      }
    }
    return false;
  }

  static void try_insert_tile(TyOp &op, PatternRewriter &rewriter, int idx,
                              int axis, int tile) {
    Value opd = op.getOperand(idx);
    auto def_op = opd.getDefiningOp();
    auto input_shape = module::getShape(opd);
    auto newType =
        RankedTensorType::get(input_shape, module::getStorageType(opd));
    auto name = module::getName(opd).str();
    if (opd && !isa<ReturnOp>(def_op)) {
      name += "_" + module::getName(op.getOperation()).str();
    }
    name += "_tile";
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> weight_tile(input_shape.size(), 1);
    weight_tile[axis] = tile;
    attrs.emplace_back(
        rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr(weight_tile)));
    auto tileOp = rewriter.create<tpu::TileOp>(
        loc, newType, ValueRange{opd, module::getNoneOp(opd.getDefiningOp())},
        attrs);
    op->setOperand(idx, tileOp);
    std::vector<int64_t> output_shape = input_shape;
    output_shape[axis] = tile;
    module::setShape(tileOp.getOutput(), output_shape);
  }

  static std::vector<int64_t>
  get_acscending_order_to_broadcast(const std::vector<int64_t> shape1,
                                    const std::vector<int64_t> shape2) {
    std::vector<int64_t> broadcast_axes;

    // to support different dims
    for (size_t i = 0; i < shape1.size(); ++i) {
      if ((shape1[i] != shape2[i]) && (shape1[i] == 1 || shape2[i] == 1)) {
        broadcast_axes.push_back(i);
      }
    }
    std::sort(broadcast_axes.begin(), broadcast_axes.end(),
              [&](int64_t a, int64_t b) {
                return std::max(shape1[a], shape2[a]) <
                       std::max(shape1[b], shape2[b]);
              });
    return broadcast_axes;
  }

  LogicalResult matchAndRewriteImpl(TyOp op,
                                    PatternRewriter &rewriter) const override {
    int max_allow_dim_backend = 4;
    Value out = op.getOutput();
    if (isa<ReturnOp>(op))
      return failure();
    int opd_num = op.getNumOperands();
    if (opd_num != 2)
      return failure();

    Value opd1 = op.getOperand(0);
    Value opd2 = op.getOperand(1);
    const std::vector<int64_t> shape1 = module::getShape(opd1);
    const std::vector<int64_t> shape2 = module::getShape(opd2);
    int shape_dim = std::max(shape1.size(), shape2.size());
    if (needBroadcast(shape1, shape2) &&
        !canMergeTo4D(shape1, shape2, shape_dim)) {

      for (int i = 0; i <= shape_dim - max_allow_dim_backend; ++i) {
        if (shape1[i] == shape2[i]) {
          continue;
        } else if (shape1[i] == 1) {
          try_insert_tile(op, rewriter, 0, i, shape2[i]);
        } else if (shape2[i] == 1) {
          try_insert_tile(op, rewriter, 1, i, shape1[i]);
        }
      }
      return success();
    }
    return failure();
  }
};

class Concat5dto4d : public OpRewriterPatternEx<tpu::ConcatOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  Concat5dto4d(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ConcatOp>(context, "Concat5dto4d", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::ConcatOp op,
                                    PatternRewriter &rewriter) const override {
    auto dims = module::getShape(op.getInputs()[0]).size();
    if (dims != 5 || op.getAxis() != 3)
      return failure();
    auto output = op.getOutput();
    auto output_shape = module::getShape(output);
    // add reshape before concat
    int i = 0;
    std::vector<Value> operands;
    // TODO: case not genaral enough, need to match all cases
    for (auto input : op.getInputs()) {
      auto input_shape = module::getShape(input);
      auto name = module::getName(input);
      if (input_shape.size() != 5 || input_shape[3] != 1) {
        return failure();
      }
      std::vector<int64_t> reshape_shape = {input_shape[0], input_shape[1],
                                            input_shape[2], input_shape[4]};
      auto reshape_type = module::getTypeLike(output, reshape_shape);
      auto loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape" +
                                                     std::to_string(i++)));
      rewriter.setInsertionPointAfterValue(input);
      auto reshape_op =
          rewriter.create<tpu::ReshapeOp>(loc, reshape_type, ValueRange{input});
      // input.replaceAllUsesWith(reshape_op.getOutput());
      operands.push_back(reshape_op.getOutput());
    }
    // renew concat
    rewriter.setInsertionPoint(op);
    // int64_t = op.getAxis();
    std::vector<NamedAttribute> new_attrs;
    // Assert(op.getAxis()>=2);
    std::vector<int64_t> concat_shape = {
        output_shape[0], output_shape[1], output_shape[2],
        output_shape[4] * static_cast<long>(op.getInputs().size())};
    auto concat_type = module::getTypeLike(output, concat_shape);
    new_attrs.emplace_back(rewriter.getNamedAttr(
        "axis", rewriter.getSI32IntegerAttr(op.getAxis())));
    auto concat_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getOutput()).str() + "_reshape_before"));
    auto new_concat_op = rewriter.create<tpu::ConcatOp>(concat_loc, concat_type,
                                                        operands, new_attrs);
    // rewriter.replaceOp(op, new_concat_op.getOutput());
    auto reshape_type = module::getTypeLike(output, output_shape);
    rewriter.setInsertionPointAfterValue(new_concat_op.getOutput());
    auto reshape_op = rewriter.create<tpu::ReshapeOp>(
        op.getLoc(), reshape_type, ValueRange{new_concat_op.getOutput()});
    // rewriter.replaceAllUsesExcept(op.getOutput(), reshape_op.getOutput(),
    // reshape_op);
    rewriter.replaceAllUsesWith(op.getOutput(), reshape_op.getOutput());
    rewriter.eraseOp(op);

    return success();
  }
};

struct ScatterElementsPattern
    : public OpRewriterPatternEx<tpu::ScatterElementsOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  ScatterElementsPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ScatterElementsOp>(
            context, "ScatterElementsPattern", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::ScatterElementsOp op,
                                    PatternRewriter &rewriter) const override {

    auto indices = op.getIndices();
    auto indices_shape = module::getShape(indices);
    auto indices_dims = indices_shape.size();
    auto axis = op.getAxis();
    if (axis == indices_dims - 1) {
      return failure();
    }
    if (!op.getIndicesCoeff().getType().isa<NoneType>()) {
      return failure();
    }
    auto input = op.getInput();
    auto input_shape = module::getShape(input);
    std::vector<Value> operands;
    operands.push_back(op.getInput());
    operands.push_back(indices);
    operands.push_back(op.getUpdates());

    auto indice_type = module::getElementType(indices);
    auto type = RankedTensorType::get(indices_shape, indice_type);
    int indices_shape8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int input_shape8[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    for (int i = 0; i < indices_dims; ++i) {
      indices_shape8[i] = indices_shape[i];
      input_shape8[i] = input_shape[i];
    }
    std::vector<int> indices_coeff;

    int tmp = 0;
    // loop for 8 times
    for (int i0 = 0; i0 < indices_shape8[0]; ++i0) {
      int tmp0 = 0;
      tmp0 += axis == 0 ? 0 : i0;
      tmp0 *= input_shape8[1];
      for (int i1 = 0; i1 < indices_shape8[1]; ++i1) {
        int tmp1 = tmp0;
        tmp1 += axis == 1 ? 0 : i1;
        tmp1 *= input_shape8[2];
        for (int i2 = 0; i2 < indices_shape8[2]; ++i2) {
          int tmp2 = tmp1;
          tmp2 += axis == 2 ? 0 : i2;
          tmp2 *= input_shape8[3];
          for (int i3 = 0; i3 < indices_shape8[3]; ++i3) {
            int tmp3 = tmp2;
            tmp3 += axis == 3 ? 0 : i3;
            tmp3 *= input_shape8[4];
            for (int i4 = 0; i4 < indices_shape8[4]; ++i4) {
              int tmp4 = tmp3;
              tmp4 += axis == 4 ? 0 : i4;
              tmp4 *= input_shape8[5];
              for (int i5 = 0; i5 < indices_shape8[5]; ++i5) {
                int tmp5 = tmp4;
                tmp5 += axis == 5 ? 0 : i5;
                tmp5 *= input_shape8[6];
                for (int i6 = 0; i6 < indices_shape8[6]; ++i6) {
                  int tmp6 = tmp5;
                  tmp6 += axis == 6 ? 0 : i6;
                  tmp6 *= input_shape8[7];
                  for (int i7 = 0; i7 < indices_shape8[7]; ++i7) {
                    tmp++;
                    int tmp7 = tmp6;
                    tmp7 += i7;
                    indices_coeff.push_back(tmp7);
                    // llvm::outs() << tmp << " " << tmp7 << "\n";
                  }
                }
              }
            }
          }
        }
      }
    }

    auto indices_coeff_op =
        top::WeightOp::create(op, "indices_coeff", indices_coeff, type);
    operands.push_back(indices_coeff_op);

    operands.push_back(op.getBuffer());
    rewriter.setInsertionPointAfter(op);
    auto new_op = rewriter.create<tpu::ScatterElementsOp>(
        op.getLoc(), op.getResult().getType(), operands, op->getAttrs());
    op.getOutput().replaceAllUsesWith(new_op.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

// permute + (mulconst) + add + cast + softmax + cast + permute
// -> add + cast + softmax + cast
struct PermuteFuseAddSoftmax : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  PermuteFuseAddSoftmax(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteFuseAddSoftmax",
                                            benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto out = op->getResult(0);
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (!op->hasOneUse()) {
      return failure();
    }
    Operation *cast_bottom_op = nullptr;
    Operation *cast_top_op = nullptr;
    auto softmax_op = dyn_cast<tpu::SoftmaxOp>(in.getDefiningOp());
    Operation *pre_op = softmax_op;
    if (!softmax_op) {
      cast_bottom_op = in.getDefiningOp();
      if (!isa<tpu::CastOp>(cast_bottom_op)) {
        return failure();
      }
      softmax_op = dyn_cast<tpu::SoftmaxOp>(
          cast_bottom_op->getOperand(0).getDefiningOp());
      if (!softmax_op) {
        return failure();
      }
      cast_top_op = softmax_op->getOperand(0).getDefiningOp();
      if (!isa<tpu::CastOp>(cast_top_op)) {
        return failure();
      }
      pre_op = cast_top_op;
    }

    auto add_op = dyn_cast<tpu::AddOp>(pre_op->getOperand(0).getDefiningOp());
    if (!add_op) {
      return failure();
    }
    auto mul_const_op =
        dyn_cast<tpu::MulConstOp>(add_op->getOperand(0).getDefiningOp());
    auto permute_op =
        dyn_cast<tpu::PermuteOp>(add_op->getOperand(0).getDefiningOp());
    if (mul_const_op) {
      permute_op =
          dyn_cast<tpu::PermuteOp>(mul_const_op->getOperand(0).getDefiningOp());
    }
    if (!permute_op) {
      return failure();
    }
    auto top_order = module::getI64Array(permute_op.getOrder());
    auto bottom_order = module::getI64Array(op.getOrder());
    if (false == (top_order->size() == 4 && top_order->at(0) == 0 &&
                  top_order->at(1) == 2 && top_order->at(2) == 1 &&
                  top_order->at(3) == 3)) {
      return failure();
    }
    if (false == (bottom_order->size() == 4 && bottom_order->at(0) == 0 &&
                  bottom_order->at(1) == 2 && bottom_order->at(2) == 1 &&
                  bottom_order->at(3) == 3)) {
      return failure();
    }
    // Define Param
    auto ori_shape = module::getShape(out);
    auto mask_shape = module::getShape(add_op->getOperand(1));
    auto mask_name = module::getName(add_op->getOperand(1)).str();
    auto add_name = module::getName(add_op.getOutput());
    if (mask_shape[1] != 1) {
      return failure();
    }

    // MulConstOp
    if (mul_const_op) {
      module::setShape(mul_const_op->getOperand(0), ori_shape);
      module::setShape(mul_const_op->getResult(0), ori_shape);
    }
    // AddOp
    module::setShape(add_op->getOperand(0), ori_shape);
    module::setShape(add_op->getResult(0), ori_shape);
    // CastOp
    if (cast_top_op) {
      module::setShape(cast_top_op->getOperand(0), ori_shape);
      module::setShape(cast_top_op->getResult(0), ori_shape);
    }
    // SoftmaxOp
    module::setShape(softmax_op->getOperand(0), ori_shape);
    module::setShape(softmax_op->getResult(0), ori_shape);
    // CastOp
    if (cast_bottom_op) {
      module::setShape(cast_bottom_op->getOperand(0), ori_shape);
      module::setShape(cast_bottom_op->getResult(0), ori_shape);
    }

    // AddOp
    rewriter.setInsertionPoint(add_op);
    auto out_add = add_op->getResult(0);
    auto new_mask_type = RankedTensorType::get(
        {mask_shape[0], mask_shape[2], mask_shape[1], mask_shape[3]},
        module::getElementType(out_add));
    if (mask_shape[1] == 1 && mask_shape[2] == 1) {
      // nothing to do
    } else {
      auto reshape_op = rewriter.create<tpu::ReshapeOp>(
          NameLoc::get(rewriter.getStringAttr(mask_name + "_" + add_name)),
          new_mask_type, add_op->getOperand(1));
      add_op->setOperand(1, reshape_op->getResult(0));
    }
    if (mul_const_op) {
      mul_const_op->setOperand(0, permute_op->getOperand(0));
    } else {
      add_op->setOperand(0, permute_op->getOperand(0));
    }
    rewriter.eraseOp(permute_op);

    // PermuteOp
    auto next_op = *op->getResult(0).user_begin();
    next_op->setOperand(0, op->getOperand(0));
    rewriter.eraseOp(op);
    return success();
  }
};

// permute + (mulconst) + add + cast + softmax + cast + slice + permute
// -> add + cast + softmax + cast + slice
struct PermuteFuseAddSoftmaxSlice : public OpRewriterPatternEx<tpu::ReshapeOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  PermuteFuseAddSoftmaxSlice(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ReshapeOp>(
            context, "PermuteFuseAddSoftmaxSlice", benifit) {}

  tpu::ReshapeOp move_reshape_after_add(tpu::AddOp &op,
                                        PatternRewriter &rewriter) const {
    auto l_reshape_op = op.getOperand(0).getDefiningOp<tpu::ReshapeOp>();
    auto r_reshape_op = op.getOperand(1).getDefiningOp<tpu::ReshapeOp>();
    if (!l_reshape_op || !r_reshape_op)
      return nullptr;
    auto l_in_shape = module::getShape(l_reshape_op.getInput()).vec();
    auto r_in_shape = module::getShape(r_reshape_op.getInput()).vec();
    if (l_in_shape != r_in_shape)
      return nullptr;
    auto l_out_shape = module::getShape(l_reshape_op.getOutput()).vec();
    auto r_out_shape = module::getShape(r_reshape_op.getOutput()).vec();
    if (l_out_shape != r_out_shape)
      return nullptr;
    auto loc = op.getLoc();
    op.setOperand(0, l_reshape_op.getInput());
    op.setOperand(1, r_reshape_op.getInput());
    auto output = op.getOutput();
    module::setShape(output, l_in_shape);
    module::setLocSuffix(op, "before_reshape");

    rewriter.setInsertionPointAfterValue(output);
    auto reshape_type = module::getTypeLike(output, l_out_shape);
    auto new_reshape_op =
        rewriter.create<tpu::ReshapeOp>(loc, reshape_type, ValueRange{output});
    rewriter.replaceAllUsesExcept(output, new_reshape_op.getOutput(),
                                  new_reshape_op);
    return new_reshape_op;
  }

  std::vector<int64_t> modify_top_matmul(tpu::MatMulOp &op,
                                         tpu::MulConstOp &mulconst_op,
                                         Value &left, Value &right,
                                         PatternRewriter &rewriter) const {
    op.setOperand(0, left);
    op.setOperand(1, right);
    auto matmul_output_shape = module::getShape(op.getOutput()).vec();
    int64_t batch_size = matmul_output_shape[0];
    int64_t num_head = matmul_output_shape[1];
    int64_t seq_length = matmul_output_shape[3];
    std::vector<int64_t> new_shape = {batch_size, 1, num_head, seq_length};
    module::setShape(op->getResult(0), {batch_size, 1, num_head, seq_length});
    module::setShape(mulconst_op->getOperand(0), new_shape);
    module::setShape(mulconst_op->getResult(0), new_shape);
    op->setAttr("hdim_is_batch", rewriter.getBoolAttr(true));
    op->setAttr("right_transpose", rewriter.getBoolAttr(true));

    auto add_op =
        dyn_cast<tpu::AddOp>(*mulconst_op.getOutput().getUsers().begin());
    if (add_op) {
      module::setShape(add_op->getOperand(0), new_shape);
      module::setShape(add_op->getResult(0), new_shape);
    }
    return new_shape;
  }

  std::vector<int64_t> modify_matmul(tpu::MatMulOp &op,
                                     PatternRewriter &rewriter) const {
    // auto permute_input_shape =
    // module::getShape(permute_op->getOperand(0)).vec();
    auto output_shape = module::getShape(op.getOutput()).vec();
    auto new_out_shape = {output_shape[0], output_shape[2], output_shape[1],
                          output_shape[3]};
    // module::setShape(op->getOperand(0), permute_input_shape);
    module::setShape(op->getResult(0), new_out_shape);
    op->setAttr("hdim_is_batch", rewriter.getBoolAttr(true));
    return new_out_shape;
  }

  template <typename TyOp>
  void modify_shape(TyOp &op, std::vector<std::vector<int64_t>> in_shapes,
                    std::vector<std::vector<int64_t>> out_shapes,
                    PatternRewriter &rewriter) const {
    for (size_t i = 0; i < in_shapes.size(); i++) {
      module::setShape(op->getOperand(i), in_shapes[i]);
    }
    for (size_t i = 0; i < out_shapes.size(); i++) {
      module::setShape(op->getResult(i), out_shapes[i]);
    }
    return;
  }

  template <>
  void modify_shape(tpu::SliceOp &op,
                    std::vector<std::vector<int64_t>> in_shapes,
                    std::vector<std::vector<int64_t>> out_shapes,
                    PatternRewriter &rewriter) const {
    for (size_t i = 0; i < in_shapes.size(); i++) {
      module::setShape(op->getOperand(i), in_shapes[i]);
    }
    for (size_t i = 0; i < out_shapes.size(); i++) {
      module::setShape(op->getResult(i), out_shapes[i]);
    }
    auto ends = module::getI64Array(op.getEnds());
    std::vector<int64_t> ends_v(ends->data(), ends->data() + ends->size());
    op->setAttr("ends", rewriter.getI64ArrayAttr(
                            {ends_v[0], ends_v[2], ends_v[1], ends_v[3]}));
    return;
  }

  LogicalResult matchAndRewriteImpl(tpu::ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto add_op = dyn_cast<tpu::AddOp>(op->getOperand(0).getDefiningOp());
    if (!add_op) {
      return failure();
    }
    int opd_num = add_op->getNumOperands();
    if (opd_num != 2)
      return failure();

    // 1. Match Pattern
    // share branch & unshare branch & self branch
    auto share_unshare_branch =
        dyn_cast<tpu::AddOp>(add_op->getOperand(0).getDefiningOp());
    auto self_branch =
        dyn_cast<tpu::MatMulOp>(add_op->getOperand(1).getDefiningOp());
    if (!share_unshare_branch || !self_branch) {
      return failure();
    }
    auto share_branch = dyn_cast<tpu::MatMulOp>(
        share_unshare_branch->getOperand(0).getDefiningOp());
    auto unshare_branch = dyn_cast<tpu::MatMulOp>(
        share_unshare_branch->getOperand(1).getDefiningOp());
    if (!share_branch || !unshare_branch) {
      return failure();
    }

    // share branch
    auto share_slice_op =
        dyn_cast<tpu::SliceOp>(share_branch->getOperand(0).getDefiningOp());
    auto share_tile_op =
        dyn_cast<tpu::TileOp>(share_branch->getOperand(1).getDefiningOp());
    if (!share_slice_op || !share_tile_op) {
      return failure();
    }
    auto share_permute_op =
        dyn_cast<tpu::PermuteOp>(share_tile_op->getOperand(0).getDefiningOp());
    if (!share_permute_op) {
      return failure();
    }

    // unshare branch
    auto unshare_slice_op =
        dyn_cast<tpu::SliceOp>(unshare_branch->getOperand(0).getDefiningOp());
    auto unshare_permute_op =
        dyn_cast<tpu::PermuteOp>(unshare_branch->getOperand(1).getDefiningOp());
    if (!unshare_slice_op || !unshare_permute_op) {
      return failure();
    }

    auto softmax_op = dyn_cast<tpu::SoftmaxOp>(
        unshare_slice_op->getOperand(0).getDefiningOp());
    if (!softmax_op) {
      return failure();
    }

    // self branch
    auto self_slice_op =
        dyn_cast<tpu::SliceOp>(self_branch->getOperand(0).getDefiningOp());
    auto _self_reshape_op = self_branch->getOperand(1).getDefiningOp();
    tpu::ReshapeOp self_reshape_op;
    if (isa<tpu::CastOp>(_self_reshape_op)) {
      self_reshape_op = dyn_cast<tpu::ReshapeOp>(
          _self_reshape_op->getOperand(0).getDefiningOp());
    } else {
      self_reshape_op = dyn_cast<tpu::ReshapeOp>(_self_reshape_op);
    }
    if (!self_slice_op || !self_reshape_op) {
      return failure();
    }

    // Softmax
    auto concat_op =
        dyn_cast<tpu::ConcatOp>(softmax_op->getOperand(0).getDefiningOp());
    if (!concat_op) {
      return failure();
    }
    auto share_add_op =
        dyn_cast<tpu::AddOp>(concat_op->getOperand(0).getDefiningOp());
    auto unshare_add_op =
        dyn_cast<tpu::AddOp>(concat_op->getOperand(1).getDefiningOp());
    auto self_mulconst_op =
        dyn_cast<tpu::MulConstOp>(concat_op->getOperand(2).getDefiningOp());
    if (!share_add_op || !unshare_add_op || !self_mulconst_op) {
      return failure();
    }

    // share branch & unshare branch
    // MulConst
    auto share_mulconst_op =
        dyn_cast<tpu::MulConstOp>(share_add_op->getOperand(0).getDefiningOp());
    auto unshare_mulconst_op = dyn_cast<tpu::MulConstOp>(
        unshare_add_op->getOperand(0).getDefiningOp());
    if (!share_mulconst_op || !unshare_mulconst_op) {
      return failure();
    }

    // MatMul
    auto share_top_matmul_op = dyn_cast<tpu::MatMulOp>(
        share_mulconst_op->getOperand(0).getDefiningOp());
    auto unshare_top_matmul_op = dyn_cast<tpu::MatMulOp>(
        unshare_mulconst_op->getOperand(0).getDefiningOp());
    auto self_top_matmul_op = dyn_cast<tpu::MatMulOp>(
        self_mulconst_op->getOperand(0).getDefiningOp());
    if (!share_top_matmul_op || !unshare_top_matmul_op || !self_top_matmul_op) {
      return failure();
    }

    // share branch
    auto top_add_op = dyn_cast<tpu::AddOp>(
        share_top_matmul_op->getOperand(0).getDefiningOp());
    auto share_top_tile_op = dyn_cast<tpu::TileOp>(
        share_top_matmul_op->getOperand(1).getDefiningOp());
    if (!top_add_op || !share_top_tile_op) {
      return failure();
    }
    auto share_top_permute_op = dyn_cast<tpu::PermuteOp>(
        share_top_tile_op->getOperand(0).getDefiningOp());
    if (!share_top_permute_op) {
      return failure();
    }

    // unshare branch
    auto unshare_top_permute_op = dyn_cast<tpu::PermuteOp>(
        unshare_top_matmul_op->getOperand(1).getDefiningOp());
    if (!unshare_top_permute_op) {
      return failure();
    }

    // self branch
    auto self_top_permute_op = dyn_cast<tpu::PermuteOp>(
        self_top_matmul_op->getOperand(1).getDefiningOp());
    if (!self_top_permute_op) {
      return failure();
    }

    // move reshape
    auto new_top_reshape_op = move_reshape_after_add(top_add_op, rewriter);

    // 2. Set Shape & Operand
    // share branch
    // TileOp
    auto share_top_permute_input_shape =
        module::getShape(share_top_permute_op->getOperand(0));
    auto share_top_tile_output_shape =
        module::getShape(share_top_tile_op->getResult(0));
    std::vector<int64_t> new_share_top_tile_output_shape =
        share_top_permute_input_shape;
    new_share_top_tile_output_shape[0] = share_top_tile_output_shape[0];
    module::setShape(share_top_tile_op->getResult(0),
                     new_share_top_tile_output_shape);
    share_top_tile_op->setOperand(0, share_top_permute_op->getOperand(0));
    // ReshapeOp
    Value top_reshape_input = new_top_reshape_op->getOperand(0);
    // Value share_top_tile_output = share_top_tile_op->getResult(0);
    // auto share_top_shape = modify_top_matmul(share_top_matmul_op,
    // share_mulconst_op, top_reshape_input, share_top_tile_output, rewriter);
    Value share_top_permute_input = share_top_permute_op->getOperand(0);
    auto share_top_shape =
        modify_top_matmul(share_top_matmul_op, share_mulconst_op,
                          top_reshape_input, share_top_permute_input, rewriter);

    // unshare branch
    Value unshare_top_permute_input = unshare_top_permute_op->getOperand(0);
    auto unshare_top_shape = modify_top_matmul(
        unshare_top_matmul_op, unshare_mulconst_op, top_reshape_input,
        unshare_top_permute_input, rewriter);

    // self branch
    Value self_top_permute_input = self_top_permute_op->getOperand(0);
    auto self_top_shape =
        modify_top_matmul(self_top_matmul_op, self_mulconst_op,
                          top_reshape_input, self_top_permute_input, rewriter);

    // ConcatOp
    auto concat_shape = module::getShape(concat_op->getResult(0)).vec();
    std::vector<int64_t> new_concat_shape = {concat_shape[0], concat_shape[2],
                                             concat_shape[1], concat_shape[3]};
    modify_shape<tpu::ConcatOp>(
        concat_op, {share_top_shape, unshare_top_shape, self_top_shape},
        {new_concat_shape}, rewriter);

    // SoftmaxOp
    modify_shape<tpu::SoftmaxOp>(softmax_op, {new_concat_shape},
                                 {new_concat_shape}, rewriter);

    // SliceOp
    auto unshare_slice_shape =
        module::getShape(unshare_slice_op->getResult(0)).vec();
    auto share_slice_shape =
        module::getShape(share_slice_op->getResult(0)).vec();
    auto self_slice_shape = module::getShape(self_slice_op->getResult(0)).vec();
    std::vector<int64_t> new_unshare_slice_shape = {
        unshare_slice_shape[0], unshare_slice_shape[2], unshare_slice_shape[1],
        unshare_slice_shape[3]};
    std::vector<int64_t> new_share_slice_shape = {
        share_slice_shape[0], share_slice_shape[2], share_slice_shape[1],
        share_slice_shape[3]};
    std::vector<int64_t> new_self_slice_shape = {
        self_slice_shape[0], self_slice_shape[2], self_slice_shape[1],
        self_slice_shape[3]};
    modify_shape<tpu::SliceOp>(unshare_slice_op, {new_concat_shape},
                               {new_unshare_slice_shape}, rewriter);
    modify_shape<tpu::SliceOp>(share_slice_op, {new_concat_shape},
                               {new_share_slice_shape}, rewriter);
    modify_shape<tpu::SliceOp>(self_slice_op, {new_concat_shape},
                               {new_self_slice_shape}, rewriter);

    // MatMulOp
    // share branch
    auto share_permute_input_shape =
        module::getShape(share_permute_op->getOperand(0));
    auto share_tile_output_shape =
        module::getShape(share_tile_op->getResult(0));
    std::vector<int64_t> new_share_tile_output_shape =
        share_permute_input_shape;
    new_share_tile_output_shape[0] = share_tile_output_shape[0];
    module::setShape(share_tile_op->getResult(0), new_share_tile_output_shape);
    share_tile_op->setOperand(0, share_permute_op->getOperand(0));
    share_branch->setOperand(1, share_permute_op->getOperand(0));
    modify_matmul(share_branch, rewriter);

    // unshare branch
    unshare_branch->setOperand(1, unshare_permute_op->getOperand(0));
    modify_matmul(unshare_branch, rewriter);

    // self branch
    self_branch->setOperand(1, self_reshape_op->getOperand(0));
    auto new_add_shape = modify_matmul(self_branch, rewriter);

    // AddOp
    modify_shape<tpu::AddOp>(share_unshare_branch, {new_add_shape},
                             {new_add_shape}, rewriter);
    modify_shape<tpu::AddOp>(add_op, {new_add_shape}, {new_add_shape},
                             rewriter);
    return success();
  }
};

// reshape + permute -> permute
struct ReshapePermuteFuse : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  ReshapePermuteFuse(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "ReshapePermuteFuse",
                                            benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (!op->hasOneUse()) {
      return failure();
    }
    auto reshape_op = dyn_cast<tpu::ReshapeOp>(op.getInput().getDefiningOp());
    if (!reshape_op) {
      return failure();
    }
    auto order = module::getI64Array(op.getOrder());
    if (!(order->size() == 4 && order->at(0) == 0 && order->at(1) == 2 &&
          order->at(2) == 1 && order->at(3) == 3)) {
      return failure();
    }
    auto input_shape = module::getShape(in);
    if (!(input_shape[0] == 1 && input_shape[2] == 1)) {
      return failure();
    }
    auto pre_input_shape = module::getShape(reshape_op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    if (pre_input_shape != out_shape) {
      return failure();
    }

    // ReshapeOp
    op.getOutput().replaceAllUsesWith(reshape_op.getInput());
    rewriter.eraseOp(op);
    rewriter.eraseOp(reshape_op);
    return success();
  }
};

// permute + reshape -> reshape
struct PermuteReshapeFuse : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteReshapeFuse(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteReshapeFuse",
                                            benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (!op->hasOneUse()) {
      return failure();
    }
    auto reshape_op = dyn_cast<tpu::ReshapeOp>(*op->getResult(0).user_begin());
    if (!reshape_op) {
      return failure();
    }
    auto order = module::getI64Array(op.getOrder());
    if (false ==
        (order->size() == 4 && order->at(0) == 0 && order->at(1) == 2 &&
         order->at(2) == 1 && order->at(3) == 3)) {
      return failure();
    }
    auto input_shape = module::getShape(in);
    if (false == (input_shape[0] == 1 && input_shape[1] == 1)) {
      return failure();
    }
    // ReshapeOp
    module::setShape(reshape_op->getOperand(0), input_shape);
    reshape_op->setOperand(0, op->getOperand(0));
    rewriter.eraseOp(op);
    return success();
  }
};

// cast (int2float) + ... + cast (float2int) + gatherelements() -> ... +
// gatherelements
struct EliminateCastBeforeGatherElements
    : public OpRewriterPatternEx<tpu::CastOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  EliminateCastBeforeGatherElements(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::CastOp>(
            context, "EliminateCastBeforeGatherElements", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::CastOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto stype = module::getStorageType(in);
    if (!stype.isInteger(32)) {
      return failure();
    }

    auto unsqueeze_op =
        dyn_cast_or_null<tpu::UnsqueezeOp>(*op.getOutput().getUsers().begin());
    if (!unsqueeze_op)
      return failure();
    auto tile_op = dyn_cast_or_null<tpu::TileOp>(
        *unsqueeze_op.getOutput().getUsers().begin());
    if (!tile_op) {
      return failure();
    }
    auto cast_op =
        dyn_cast_or_null<tpu::CastOp>(*tile_op.getOutput().getUsers().begin());
    if (!cast_op ||
        !module::getStorageType(cast_op->getResult(0)).isInteger(32)) {
      return failure();
    }

    unsqueeze_op->setOperand(0, in);
    auto unsqueeze_shape = module::getShape(unsqueeze_op->getResult(0));
    auto unsqueeze_new_type = module::getTypeLike(in, unsqueeze_shape);
    unsqueeze_op->getResult(0).setType(unsqueeze_new_type);

    auto tile_shape = module::getShape(tile_op->getResult(0));
    auto tile_new_shape =
        module::getTypeLike(tile_op->getOperand(0), tile_shape);
    tile_op->getResult(0).setType(tile_new_shape);
    cast_op->getResult(0).replaceAllUsesWith(tile_op->getResult(0));
    rewriter.eraseOp(op);
    rewriter.eraseOp(cast_op);
    return success();
  }
};
} // namespace bm1684x

//  reshape + permute + reshape + permute -> reshape + permute
//            3D(0,2,1) 6D        6D case1:(0,2,4,3,5,1)
//                                   case2:(0,2,4,1,3,5)
struct PermuteReshapeFuse2 : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  PermuteReshapeFuse2(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "PermuteReshapeFuse2",
                                            benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();

    int pattern_case = -1;
    // return failure();
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (!op->hasOneUse()) {
      return failure();
    }
    auto reshape_op = dyn_cast<tpu::ReshapeOp>(*op->getResult(0).user_begin());
    if (!reshape_op) {
      return failure();
    }

    auto permute_op =
        dyn_cast<tpu::PermuteOp>(*reshape_op->getResult(0).user_begin());
    if (!permute_op) {
      return failure();
    }

    if (!reshape_op->hasOneUse()) {
      return failure();
    }
    auto op_shape = module::getShape(op);

    auto shape = module::getShape(reshape_op);
    if (false == (shape.size() == 6 &&
                  (shape[2] * shape[3] * shape[4] * shape[5] == op_shape[2]))) {
      return failure();
    }

    auto order = module::getI64Array(permute_op.getOrder());
    if (order->size() == 6 && order->at(0) == 0 && order->at(1) == 2 &&
        order->at(2) == 4 && order->at(3) == 3 && order->at(4) == 5 &&
        order->at(5) == 1) {
      pattern_case = 1;
    }
    if (order->size() == 6 && order->at(0) == 0 && order->at(1) == 2 &&
        order->at(2) == 4 && order->at(3) == 1 && order->at(4) == 3 &&
        order->at(5) == 5) {
      pattern_case = 2;
    }
    if (pattern_case < 0) {
      return failure();
    }

    std::vector<int64_t> new_shape = shape;
    // ReshapeOp
    new_shape[0] = shape[0];
    new_shape[1] = shape[2];
    new_shape[2] = shape[3];
    new_shape[3] = shape[4];
    new_shape[4] = shape[5];
    new_shape[5] = shape[1];
    auto loc = module::getLocLike(op.getOutput(), "Reshape");
    rewriter.setInsertionPoint(op);
    auto rs_op = rewriter.create<tpu::ReshapeOp>(
        loc, reshape_op.getOutput().getType(), ValueRange{op.getInput()});
    reshape_op.getOutput().replaceAllUsesWith(rs_op.getOutput());

    module::setShape(rs_op, new_shape);
    rewriter.eraseOp(reshape_op);
    rewriter.eraseOp(op);
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> new_order;
    if (pattern_case == 1) {
      new_order = {0, 1, 3, 2, 4, 5};
    }
    if (pattern_case == 2) {
      new_order = {0, 1, 3, 5, 2, 4};
    }
    permute_op->setAttr("order", rewriter.getI64ArrayAttr(new_order));
    return success();
  }
};

struct FitPermute2Hdim : public OpRewriterPatternEx<tpu::MatMulOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  FitPermute2Hdim(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "FitPermute2Hdim",
                                           benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {

    int pattern_case = -1;
    auto left = op.getInput();
    auto right = op.getRight();
    auto l_is_weight = module::isWeight(left);
    auto r_is_weight = module::isWeight(right);
    if (l_is_weight && r_is_weight) {
      return failure();
    }
    auto l_op = left.getDefiningOp();
    auto r_op = right.getDefiningOp();

    auto l_permute_op = dyn_cast<tpu::PermuteOp>(l_op);
    auto r_reshape_op = dyn_cast<tpu::ReshapeOp>(r_op);

    if (l_permute_op && r_reshape_op) {
      pattern_case = 1;
    }
    auto l_reshape_op = dyn_cast<tpu::ReshapeOp>(l_op);
    auto r_permute_op = dyn_cast<tpu::PermuteOp>(r_op);

    if (l_reshape_op && r_permute_op) {
      pattern_case = 2; // to do
    }

    if (pattern_case < 0) {
      return failure();
    }
    if (pattern_case == 1) {

      auto order = module::getI64Array(l_permute_op.getOrder());
      if (false ==
          (order->size() == 4 && order->at(0) == 0 && order->at(1) == 2 &&
           order->at(2) == 1 && order->at(3) == 3)) {
        return failure();
      }
      // check forward
      // slice(slc) -- reshape(ori_rs_op) -- permute(in_pm_op) --
      // reshape(in_rs_op) -- reshape(r_reshape_op) -- matmul(op)
      //                                                               |
      //                                                            conv2d ...

      auto in_rs = r_reshape_op.getInput();
      auto in_rs_op = dyn_cast<tpu::ReshapeOp>(in_rs.getDefiningOp());
      if (!in_rs_op) {
        return failure();
      }
      if (in_rs_op->hasOneUse()) {
        return failure();
      }

      auto in_pm = in_rs_op.getInput();
      auto in_pm_op = dyn_cast<tpu::PermuteOp>(in_pm.getDefiningOp());
      if (!in_pm_op) {
        return failure();
      }
      if (!in_pm_op->hasOneUse()) {
        return failure();
      }

      auto ori_rs = in_pm_op.getInput();
      auto ori_rs_op = dyn_cast<tpu::ReshapeOp>(ori_rs.getDefiningOp());
      if (!in_rs_op) {
        return failure();
      }
      if (!ori_rs_op->hasOneUse()) {
        return failure();
      }
      auto ori_shape = module::getShape(ori_rs_op);

      auto in_pm_order = module::getI64Array(in_pm_op.getOrder());
      if (false == (in_pm_order->size() == 6 && in_pm_order->at(0) == 0 &&
                    in_pm_order->at(1) == 1 && in_pm_order->at(2) == 3 &&
                    in_pm_order->at(3) == 5 && in_pm_order->at(4) == 2 &&
                    in_pm_order->at(5) == 4)) {
        return failure();
      }

      auto slc = ori_rs_op.getInput();
      auto slc_op = dyn_cast<tpu::SliceOp>(slc.getDefiningOp());
      if (!slc_op) {
        return failure();
      }
      if (!slc_op->hasOneUse()) {
        return failure();
      }

      // check over

      // set new right permute op order
      std::vector<int64_t> new_order = {0, 2, 3, 1};

      // set new right reshape op shape
      auto shape = module::getShape(r_reshape_op);
      std::vector<int64_t> new_shape(4);
      new_shape[0] = shape[0];
      new_shape[1] = shape[3];
      new_shape[2] = shape[1];
      new_shape[3] = shape[2];

      /**
       *                                                             conv2d ...
       *                                                                |
       * slice(slc) -- reshape(ori_rs_op) -- permute(in_pm_op) --
       * reshape(in_rs_op) -- reshape(r_reshape_op) => matmul(op) <=
       * permute(l_permute_op)
       *
       * transformed into:
       *
       * slice(slc) -- reshape(ori_rs_op) -- ...
       *                   |                                            =>
       * matmul(op) <= permute(l_permute_op) permute     --    reshape     --
       * permute   / (new_permute_op_1)  (new_reshape_op)  (new_permute_op_2)
       */
      if (ori_shape[1] == 1) {
        rewriter.setInsertionPointAfter(ori_rs_op);
        std::vector<int64_t> first_permute_order = {0, 1, 3, 2, 4, 5};

        auto new_permute_op_1_loc =
            module::getLocLike(ori_rs_op.getOutput(), "new_Transpose_1");
        auto new_permute_op_1 = rewriter.create<tpu::PermuteOp>(
            new_permute_op_1_loc, ori_rs_op.getOutput().getType(),
            ValueRange{ori_rs_op.getOutput(), module::getNoneOp(ori_rs_op)});
        new_permute_op_1->setAttr(
            "order", rewriter.getI64ArrayAttr(first_permute_order));

        rewriter.setInsertionPointAfter(new_permute_op_1);
        auto new_reshape_op_loc =
            module::getLocLike(ori_rs_op.getOutput(), "new_Reshape");
        auto new_reshape_op = rewriter.create<tpu::ReshapeOp>(
            new_reshape_op_loc, new_permute_op_1.getOutput().getType(),
            new_permute_op_1.getOutput());
        new_reshape_op.setOperand(0, new_permute_op_1.getOutput());
        module::setShape(new_reshape_op.getOutput(), new_shape);

        rewriter.setInsertionPointAfter(new_reshape_op);
        auto new_permute_op_2_loc =
            module::getLocLike(ori_rs_op.getOutput(), "new_Transpose_2");
        auto new_permute_op_2 = rewriter.create<tpu::PermuteOp>(
            new_permute_op_2_loc, r_reshape_op.getOutput().getType(),
            ValueRange{new_reshape_op.getOutput(),
                       module::getNoneOp(new_reshape_op)});
        new_permute_op_2->setAttr("order", rewriter.getI64ArrayAttr(new_order));

        op.setOperand(1, new_permute_op_2.getOutput());
        rewriter.eraseOp(r_reshape_op);

        return success();
      } else {
        // ReshapeOp
        rewriter.setInsertionPointAfter(slc_op);
        auto new_reshape_loc =
            module::getLocLike(slc_op.getOutput(), "Reshape");
        auto rs_op = rewriter.create<tpu::ReshapeOp>(
            new_reshape_loc, slc_op.getOutput().getType(), slc_op.getOutput());
        rs_op.setOperand(0, slc_op.getOutput());
        module::setShape(rs_op.getOutput(), new_shape);

        rewriter.setInsertionPointAfter(rs_op);
        auto new_permute_loc =
            module::getLocLike(rs_op.getOutput(), "Transpose");
        auto new_permute_op = rewriter.create<tpu::PermuteOp>(
            new_permute_loc, r_reshape_op.getOutput().getType(),
            ValueRange{rs_op.getOutput(), module::getNoneOp(rs_op)});
        new_permute_op->setAttr("order", rewriter.getI64ArrayAttr(new_order));

        op.setOperand(1, new_permute_op.getOutput());
        rewriter.eraseOp(r_reshape_op);

        return success();
      }
    }
    return failure();
  }
};

/**
 * permute \                            \
 *          => Add ->permute =>         -> Add ->
 * permute /                    permute /
 */

struct ErasePermuteAroundAdd : public OpRewriterPatternEx<tpu::AddOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  ErasePermuteAroundAdd(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::AddOp>(context, "ErasePermuteAroundAdd",
                                        benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::AddOp op,
                                    PatternRewriter &rewriter) const override {

    auto l_permute_op = op.getOperand(0).getDefiningOp<tpu::PermuteOp>();
    auto r_permute_op = op.getOperand(1).getDefiningOp<tpu::PermuteOp>();
    if (!l_permute_op || !r_permute_op) {
      return failure();
    }
    auto l_in_shape = module::getShape(l_permute_op.getInput()).vec();
    auto r_in_shape = module::getShape(r_permute_op.getInput()).vec();

    if (l_in_shape.size() != r_in_shape.size()) {
      return failure();
    }

    std::vector<int64_t> new_order(l_in_shape.size());

    auto next_op = *op->getResult(0).getUsers().begin();
    auto out_permute_op = dyn_cast<tpu::PermuteOp>(next_op);

    if (!out_permute_op) {
      return failure();
    }
    if (!out_permute_op->hasOneUse()) {
      return failure();
    }

    auto l_permute_order = *module::getI64Array(l_permute_op.getOrder());
    auto r_permute_order = *module::getI64Array(r_permute_op.getOrder());
    auto out_permute_order = *module::getI64Array(out_permute_op.getOrder());
    if (l_permute_order != out_permute_order &&
        r_permute_order != out_permute_order) {
      return failure();
    }

    if (r_permute_order == out_permute_order) {
      for (int i = 0; i < l_in_shape.size(); i++) {
        new_order[i] = l_permute_order[r_permute_order[i]];
      }
      op.setOperand(0, r_permute_op.getInput());
      rewriter.replaceAllUsesExcept(out_permute_op.getOutput(), op.getOutput(),
                                    op);

      l_permute_op->setAttr("order", rewriter.getI64ArrayAttr(new_order));
      auto new_shape = module::getShape(out_permute_op.getOutput()).vec();
      module::setShape(op.getOutput(), new_shape);
      module::setShape(l_permute_op.getOutput(), new_shape);
      rewriter.eraseOp(r_permute_op);
      rewriter.eraseOp(out_permute_op);
      return success();
    }

    if (l_permute_order == out_permute_order) {
      for (int i = 0; i < l_in_shape.size(); i++) {
        new_order[i] = r_permute_order[l_permute_order[i]];
      }
      op.setOperand(0, l_permute_op.getInput());
      rewriter.replaceAllUsesExcept(out_permute_op.getOutput(), op.getOutput(),
                                    op);

      r_permute_op->setAttr("order", rewriter.getI64ArrayAttr(new_order));
      auto new_shape = module::getShape(out_permute_op.getOutput()).vec();
      module::setShape(op.getOutput(), new_shape);
      module::setShape(r_permute_op.getOutput(), new_shape);
      rewriter.eraseOp(l_permute_op);
      rewriter.eraseOp(out_permute_op);
      return success();
    }
    return failure();
  }
};

/**
 * A ---------------------------------\
 *                                     => MatMulHidmBatch => ...
 * B -- Reshape2 -- Tile -- Reshape1  /
 *
 * NOTE: This is typical for Group-Query-Attention(GQA) and B is Key or Value
 *
 */
class TileMatMulHdimBatchPattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  TileMatMulHdimBatchPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(
            context, "TileMatMulHdimBatchPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {

    auto left = op.getInput();
    auto right = op.getRight();

    auto stype = module::getStorageType(left);
    if (stype.isF32() || stype.isInteger(8)) {
      return failure();
    }
    auto l_is_weight = module::isWeight(left);
    auto r_is_weight = module::isWeight(right);
    if (l_is_weight && r_is_weight) {
      return failure();
    }

    if (!l_is_weight && !r_is_weight) {
      auto r_reshape1_op = dyn_cast<tpu::ReshapeOp>(right.getDefiningOp());
      if (!(r_reshape1_op && r_reshape1_op->hasOneUse())) {
        return failure();
      }
      auto r_reshape1_input = r_reshape1_op.getInput();

      auto tile_op = dyn_cast<tpu::TileOp>(r_reshape1_input.getDefiningOp());
      if (!(tile_op && tile_op->hasOneUse())) {
        return failure();
      }
      auto tile_input = tile_op.getInput();

      auto r_reshape2_op = dyn_cast<tpu::ReshapeOp>(tile_input.getDefiningOp());
      if (!(r_reshape2_op && r_reshape2_op->hasOneUse())) {
        return failure();
      }
      auto r_reshape2_input = r_reshape2_op.getInput();
      auto shape = module::getShape(r_reshape2_input);
      // num_head of Key/Value must be 1 to do broadcast
      if (shape[2] != 1) {
        return failure();
      }
      auto hdim_is_batch = op.getHdimIsBatch();
      if (hdim_is_batch == false) {
        return failure();
      }

      r_reshape1_op.replaceAllUsesWith(r_reshape1_input);
      tile_op.replaceAllUsesWith(tile_input);
      r_reshape2_op.replaceAllUsesWith(r_reshape2_input);

      // op->setAttr("hdim_is_batch",
      // rewriter.getBoolAttr(!hdim_is_batch));
      return success();
    }
    return failure();
  }
};

#if 0
/* for to reduce the data move, mark on the Redundancy SliceOp if match below pattern:
          /--->SliceOp
   reshape---->SliceOp
         \---->SliceOp
      */
class MarkRedundancySlicePattern : public OpRewriterPatternEx<tpu::SliceOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  LogicalResult matchAndRewriteImpl(tpu::SliceOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<tpu::ReshapeOp>(op.getInput().getDefiningOp())) {
      return failure();
    }

    auto srcOp = op.getInput().getDefiningOp();
    for (Operation *user: srcOp->getUsers()) {
      if (!isa<tpu::SliceOp>(user))
        return failure();
    }
    //indicate don;t codegen later
    op->setAttr("discard", rewriter.getBoolAttr(true));
  }
};
#endif

#if 1
//  split the pattern if batch=1
class MatMulActiveMatMulPattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  MatMulActiveMatMulPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMulActiveMatMulPattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {

    auto left0 = op.getInput();
    auto right0 = op.getRight();
    auto stype = module::getStorageType(left0);
    auto mm0_left_shape = module::getShape(left0);
    if (!isa<Float16Type, BFloat16Type>(stype) ||
        !isa<top::WeightOp>(right0.getDefiningOp()) || mm0_left_shape[0] > 1) {
      return failure();
    }

    auto cast0 = dyn_cast<tpu::CastOp>(left0.getDefiningOp());
    if (!cast0) {
      return failure();
    }
    auto active0 = dyn_cast<tpu::ActiveOp>(cast0.getInput().getDefiningOp());
    if (!active0) {
      return failure();
    }
    auto cast1 = dyn_cast<tpu::CastOp>(active0.getInput().getDefiningOp());
    if (!cast1) {
      return failure();
    }
    auto mm1 = dyn_cast<tpu::MatMulOp>(cast1.getInput().getDefiningOp());
    if (!mm1) {
      return failure();
    }
    auto left1 = mm1.getInput();
    auto right1 = mm1.getRight();
    if (!isa<top::WeightOp>(right1.getDefiningOp())) {
      return failure();
    }
    if (!left1.hasOneUse()) {
      return failure();
    }

    // split the pattern
    std::vector<Value> operands;
    for (int i = 0; i < 2; ++i) {
      auto cur_out = left1;
      Operation *next_op = mm1.getOperation();
      auto suffix = std::to_string(i);
      next_op =
          tpu::cloneColParallelMatMul(rewriter, next_op, cur_out, 2, i, 0);
      next_op = tpu::cloneCommonOp(rewriter, next_op, cur_out, suffix);
      next_op =
          tpu::cloneRowParallelMatMul(rewriter, next_op, cur_out, 2, i, 0);
      operands.push_back(cur_out);
    }

    rewriter.setInsertionPointAfterValue(operands[0]);
    std::string suffix = std::string("add_");
    auto loc = module::getLocLike(operands[1], suffix);
    auto add = rewriter.create<tpu::AddOp>(
        loc, operands[0].getType(), mlir::ValueRange{operands[0], operands[1]});
    op.getOutput().replaceAllUsesWith(add.getOutput());
    return success();
  }
};
#endif

Operation *get_next_op(Operation *op,
                       std::vector<mlir::Operation *> &mulshifts) {
  auto next_op = *op->getResult(0).getUsers().begin();
  // if (!isa<tpu::MulShiftOp, tpu::CastOp>(next_op)) {
  //   return next_op;
  // }
  if (!isa<tpu::MulShiftOp>(next_op)) {
    return next_op;
  }
  mulshifts.emplace_back(next_op);
  // if (isa<tpu::MulShiftOp>(next_op)) {
  //   mulshifts.emplace_back(next_op);
  // }
  return *next_op->getResult(0).getUsers().begin();
}

class RotaryPosEmbPattern : public OpRewriterPatternEx<tpu::PermuteOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
public:
  RotaryPosEmbPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::PermuteOp>(context, "RotaryPosEmbPattern",
                                            benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::PermuteOp op,
                                    PatternRewriter &rewriter) const override {
    // check topo
    if (op->hasOneUse())
      return failure();
    std::vector<mlir::Operation *> permute_users;
    for (auto user : op.getOutput().getUsers()) {
      permute_users.emplace_back(user);
    }
    if (permute_users.size() != 2) {
      return failure();
    }
    auto slice_l_op = dyn_cast_or_null<tpu::SliceOp>(permute_users[1]);
    auto slice_r_op = dyn_cast_or_null<tpu::SliceOp>(permute_users[0]);
    if (!(slice_l_op && slice_r_op))
      return failure();
    if (!slice_l_op->hasOneUse() || slice_r_op->hasOneUse())
      return failure();
    std::vector<mlir::Operation *> slice_users;
    for (auto user : slice_r_op.getOutput().getUsers()) {
      slice_users.emplace_back(user);
    }
    if (slice_users.size() != 3) {
      return failure();
    }
    auto mul_0_op = dyn_cast_or_null<tpu::MulOp>(slice_users[2]);
    auto slice_1_op = dyn_cast_or_null<tpu::SliceOp>(slice_users[1]);
    auto slice_2_op = dyn_cast_or_null<tpu::SliceOp>(slice_users[0]);
    std::vector<mlir::Operation *> mulshifts;

    if (!(mul_0_op && slice_1_op && slice_2_op))
      return failure();
    auto mulconst_op = dyn_cast_or_null<tpu::MulConstOp>(
        *slice_1_op.getOutput().getUsers().begin());
    auto mulshift_op = dyn_cast_or_null<tpu::MulShiftOp>(
        *slice_1_op.getOutput().getUsers().begin());
    if ((!mulconst_op || !mulconst_op->hasOneUse()) &&
        (!mulshift_op || !mulshift_op->hasOneUse()))
      return failure();
    auto mulconst_or_mulshift_op = mulconst_op ? mulconst_op : mulshift_op;
    auto unsqueeze_0_op = dyn_cast_or_null<tpu::UnsqueezeOp>(
        *get_next_op(mulconst_or_mulshift_op, mulshifts));
    if (!unsqueeze_0_op || !unsqueeze_0_op->hasOneUse())
      return failure();
    auto unsqueeze_1_op =
        dyn_cast_or_null<tpu::UnsqueezeOp>(*get_next_op(slice_2_op, mulshifts));
    if (!unsqueeze_1_op || !unsqueeze_1_op->hasOneUse())
      return failure();
    auto concat_0_op = dyn_cast_or_null<tpu::ConcatOp>(
        *get_next_op(unsqueeze_0_op, mulshifts));
    if (!concat_0_op || !concat_0_op->hasOneUse())
      return failure();
    if (concat_0_op.getOperation() != get_next_op(unsqueeze_1_op, mulshifts))
      return failure();
    auto reshape_op =
        dyn_cast_or_null<tpu::ReshapeOp>(*get_next_op(concat_0_op, mulshifts));
    if (!reshape_op || !reshape_op->hasOneUse())
      return failure();
    auto mul_1_op =
        dyn_cast_or_null<tpu::MulOp>(*get_next_op(reshape_op, mulshifts));
    if (!mul_1_op || !mul_1_op->hasOneUse())
      return failure();
    auto add_op =
        dyn_cast_or_null<tpu::AddOp>(*get_next_op(mul_0_op, mulshifts));
    if (!add_op || !add_op->hasOneUse())
      return failure();
    if (add_op.getOperand(1) != mul_1_op.getOutput())
      return failure();
    auto concat_1_op =
        dyn_cast_or_null<tpu::ConcatOp>(*get_next_op(slice_l_op, mulshifts));
    if (!concat_1_op || !concat_1_op->hasOneUse())
      return failure();
    if (concat_1_op.getOperand(1) != add_op.getOutput())
      return failure();
    // check params
    auto order = *module::getI64Array(op.getOrder());
    std::vector<int64_t> order_0213{0, 2, 1, 3};
    if (order != order_0213)
      return failure();
    if (*module::getI64Array(slice_l_op.getSteps()) !=
        std::vector<int64_t>{1, 1, 1, 1})
      return failure();
    if (*module::getI64Array(slice_r_op.getSteps()) !=
        std::vector<int64_t>{1, 1, 1, 1})
      return failure();
    if (*module::getI64Array(slice_r_op.getOffset()) !=
        std::vector<int64_t>{0, 0, 1, 0})
      return failure();
    if (*module::getI64Array(slice_1_op.getSteps()) !=
        std::vector<int64_t>{1, 1, 1, 2})
      return failure();
    if (*module::getI64Array(slice_1_op.getOffset()) !=
        std::vector<int64_t>{0, 0, 0, 1})
      return failure();
    if (*module::getI64Array(slice_2_op.getSteps()) !=
        std::vector<int64_t>{1, 1, 1, 2})
      return failure();
    auto slice_r_outshape = module::getShape(slice_r_op.getOutput());
    int64_t h = slice_r_outshape[2];
    auto mulconst_or_mulshift_output = mulconst_or_mulshift_op->getResult(0);
    auto mulconst_or_mulshift_shape =
        module::getShape(mulconst_or_mulshift_output);
    if (h != mulconst_or_mulshift_shape[2])
      return failure();
    if (*module::getI64Array(unsqueeze_0_op.getAxes()) !=
        std::vector<int64_t>{-1})
      return failure();
    if (*module::getI64Array(unsqueeze_1_op.getAxes()) !=
        std::vector<int64_t>{-1})
      return failure();
    if ((concat_0_op.getAxis()) != 4)
      return failure();
    auto reshape_outshape = module::getShape(reshape_op.getOutput());
    if (reshape_outshape.size() != 4 || reshape_outshape[2] != h)
      return failure();
    auto mul_0_outshape = module::getShape(mul_0_op.getOutput());
    if (h != mul_0_outshape[2])
      return failure();
    auto mul_1_outshape = module::getShape(mul_1_op.getOutput());
    if (h != mul_1_outshape[2])
      return failure();
    auto add_outshape = module::getShape(add_op.getOutput());
    if (h != add_outshape[2])
      return failure();
    if ((concat_1_op.getAxis()) != 2)
      return failure();
    // get rid of this permute op
    auto output = op.getInput();
    op.getOutput().replaceAllUsesWith(output);
    auto permute_outshape = module::getShape(op.getOutput());
    rewriter.eraseOp(op);
    int64_t batch, head_n, hw, head_sz;
    batch = permute_outshape[0];
    head_n = permute_outshape[1];
    hw = permute_outshape[2];
    head_sz = permute_outshape[3];
    std::vector<int64_t> slice_l_shape{batch, 1, head_n, head_sz};
    module::setShape(slice_l_op.getOutput(), slice_l_shape);
    slice_l_op->setAttr("ends", rewriter.getI64ArrayAttr(slice_l_shape));
    std::vector<int64_t> common_shape{batch, hw - 1, head_n, head_sz};
    module::setShape(slice_r_op.getOutput(), common_shape);
    std::vector<int64_t> new_ends{batch, hw, head_n, head_sz};
    slice_r_op->setAttr("ends", rewriter.getI64ArrayAttr(new_ends));
    std::vector<int64_t> new_offsets{0, 1, 0, 0};
    slice_r_op->setAttr("offset", rewriter.getI64ArrayAttr(new_offsets));
    module::setShape(mul_0_op.getOutput(), common_shape);
    auto mul_0_weight = mul_0_op.getInputs()[1];
    if (module::isWeight(mul_0_weight)) {
      auto weight_0_shape = module::getShape(mul_0_weight);
      std::vector<int64_t> new_0_shape{weight_0_shape[0], weight_0_shape[2],
                                       weight_0_shape[1], weight_0_shape[3]};
      module::setShape(mul_0_weight, new_0_shape);
    }
    std::vector<int64_t> slice_1_shape{batch, hw - 1, head_n, head_sz / 2};
    module::setShape(slice_1_op.getOutput(), slice_1_shape);
    slice_1_op->setAttr("ends", rewriter.getI64ArrayAttr(common_shape));
    module::setShape(slice_2_op.getOutput(), slice_1_shape);
    slice_2_op->setAttr("ends", rewriter.getI64ArrayAttr(common_shape));
    module::setShape(mulconst_or_mulshift_op->getResult(0), slice_1_shape);
    std::vector<int64_t> unsqueeze_0_shape{batch, hw - 1, head_n, head_sz / 2,
                                           1};
    module::setShape(unsqueeze_0_op.getOutput(), unsqueeze_0_shape);

    // reshape_after_unsqueeze_0: 1*576*6*32*1->1*576*(6*32)*1
    std::vector<int64_t> reshape_after_unsqueeze_shape{batch, hw - 1,
                                                       head_n * head_sz / 2, 1};
    auto reshape_after_unsqueeze_0_type = RankedTensorType::get(
        reshape_after_unsqueeze_shape,
        module::getElementType(unsqueeze_0_op.getOutput()));
    auto reshape_after_unsqueeze_0_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(unsqueeze_0_op.getOutput()).str() +
        "_reshape__after_unsqueeze_0"));
    rewriter.setInsertionPointAfter(unsqueeze_0_op);
    auto reshape_after_unsqueeze_0_op = rewriter.create<tpu::ReshapeOp>(
        reshape_after_unsqueeze_0_loc, reshape_after_unsqueeze_0_type,
        ValueRange{unsqueeze_0_op.getOutput()});
    rewriter.replaceAllUsesExcept(unsqueeze_0_op.getOutput(),
                                  reshape_after_unsqueeze_0_op.getOutput(),
                                  reshape_after_unsqueeze_0_op);
    module::setShape(unsqueeze_1_op.getOutput(), unsqueeze_0_shape);
    auto reshape_after_unsqueeze_1_type = RankedTensorType::get(
        reshape_after_unsqueeze_shape,
        module::getElementType(unsqueeze_1_op.getOutput()));
    auto reshape_after_unsqueeze_1_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(unsqueeze_1_op.getOutput()).str() +
        "_reshape__after_unsqueeze_1"));
    rewriter.setInsertionPointAfter(unsqueeze_1_op);
    auto reshape_after_unsqueeze_1_op = rewriter.create<tpu::ReshapeOp>(
        reshape_after_unsqueeze_1_loc, reshape_after_unsqueeze_1_type,
        ValueRange{unsqueeze_1_op.getOutput()});
    rewriter.replaceAllUsesExcept(unsqueeze_1_op.getOutput(),
                                  reshape_after_unsqueeze_1_op.getOutput(),
                                  reshape_after_unsqueeze_1_op);
    std::vector<int64_t> concat_0_shape{batch, hw - 1, head_n * head_sz / 2, 2};
    module::setShape(concat_0_op.getOutput(), concat_0_shape);
    concat_0_op->setAttr("axis", rewriter.getSI32IntegerAttr(3));
    std::vector<int64_t> reshape_after_concat_0_shape{batch, hw - 1, head_n,
                                                      head_sz / 2, 2};
    auto reshape_after_concat_0_type =
        RankedTensorType::get(reshape_after_concat_0_shape,
                              module::getElementType(concat_0_op.getOutput()));
    auto reshape_after_concat_0_loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(concat_0_op.getOutput()).str() +
                               "_reshape__after_concat_0"));
    rewriter.setInsertionPointAfter(concat_0_op);
    auto reshape_after_concat_0_op = rewriter.create<tpu::ReshapeOp>(
        reshape_after_concat_0_loc, reshape_after_concat_0_type,
        ValueRange{concat_0_op.getOutput()});
    rewriter.replaceAllUsesExcept(concat_0_op.getOutput(),
                                  reshape_after_concat_0_op.getOutput(),
                                  reshape_after_concat_0_op);
    module::setShape(reshape_op.getOutput(), common_shape);
    module::setShape(mul_1_op.getOutput(), common_shape);
    auto mul_1_weight = mul_1_op.getInputs()[1];
    if (module::isWeight(mul_1_weight)) {
      auto weight_1_shape = module::getShape(mul_1_weight);
      std::vector<int64_t> new_1_shape{weight_1_shape[0], weight_1_shape[2],
                                       weight_1_shape[1], weight_1_shape[3]};
      module::setShape(mul_1_weight, new_1_shape);
    }
    module::setShape(add_op.getOutput(), common_shape);
    std::vector<int64_t> concat_1_shape{batch, hw, head_n, head_sz};
    module::setShape(concat_1_op.getOutput(), concat_1_shape);
    concat_1_op->setAttr("axis", rewriter.getSI32IntegerAttr(1));
    // create permute: 1x577x6x64 -> 1x6x577x64
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order_0213)));
    auto permute_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(concat_1_op.getOutput()).str() + "_permute"));
    auto permute_type = RankedTensorType::get(
        concat_1_shape, module::getElementType(concat_1_op.getOutput()));
    rewriter.setInsertionPointAfter(concat_1_op);
    auto permute_op = rewriter.create<tpu::PermuteOp>(
        permute_loc, permute_type,
        ValueRange{concat_1_op.getOutput(), module::getNoneOp(concat_1_op)},
        attrs);
    std::vector<int64_t> permute_shape{batch, head_n, hw, head_sz};
    module::setShape(permute_op.getOutput(), permute_shape);
    rewriter.replaceAllUsesExcept(concat_1_op.getOutput(),
                                  permute_op.getOutput(), permute_op);
    for (auto mulshift_op : mulshifts) {
      module::setShape(mulshift_op->getResult(0),
                       module::getShape(mulshift_op->getOperand(0)));
    }
    return success();
  }
};

/**
 *               -> Slice -> squeeze -> ...        -> Slice -> ...
 *              /                                 /
 * A -> Reshape --> Slice -> squeeze -> ... => A  --> Slice -> ...
 */
class ReshapeSliceSqueezePattern : public OpRewriterPatternEx<tpu::ReshapeOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;
public:
  ReshapeSliceSqueezePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ReshapeOp>(
            context, "ReshapeSliceSqueezePattern", benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto shape = module::getShape(in);
    auto users = std::vector<Operation *>(op->user_begin(), op->user_end());
    int idx = 0;
    LogicalResult change = failure();
    for (auto user : users) {
      if (!isa<tpu::SliceOp>(user) || !user->hasOneUse()) {
        continue;
      }
      auto next_op = *user->user_begin();
      if (!isa<tpu::SqueezeOp>(next_op) || !next_op->hasOneUse()) {
        continue;
      }
      auto slice = cast<tpu::SliceOp>(user);
      auto squeeze = cast<tpu::SqueezeOp>(next_op);
      auto offset = module::getI64Array(slice.getOffset());
      auto steps = module::getI64Array(slice.getSteps());
      auto ends = module::getI64Array(slice.getEnds());
      auto in_shape = module::getShape(slice.getInput());
      auto out_shape = module::getShape(slice.getOutput());
      int ax = -1;
      int inner_size = 1;
      for (int i = 0; i < in_shape.size(); ++i) {
        if (in_shape[i] != out_shape[i]) {
          if (ax == -1) {
            ax = i;
          } else {
            ax = -2;
          }
        }
        if (ax > -1 && i >= ax) {
          inner_size *= in_shape[i];
        }
      }
      if (ax < 0 || ax >= steps->size() || steps->at(ax) != 1 ||
          ax >= shape.size() || inner_size != shape[ax]) {
        break;
      }
      auto noneOp = module::getNoneOp(op);
      std::vector<Value> operands{in, noneOp, noneOp, noneOp, noneOp};
      std::vector<int64_t> new_offset(shape.size(), 0);
      std::vector<int64_t> new_steps(shape.size(), 1);
      std::vector<int64_t> new_ends(shape.size(), -1);
      new_offset[ax] = offset->at(ax) * inner_size / in_shape[ax];
      new_ends[ax] = inner_size / in_shape[ax] + new_offset[ax];
      std::vector<NamedAttribute> attrs;
      attrs.push_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr(new_offset)));
      attrs.push_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(new_steps)));
      attrs.push_back(
          rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(new_ends)));

      rewriter.setInsertionPointAfterValue(in);
      const std::string name_slice =
          module::getName(in).str() + "_slice_" + std::to_string(idx++);
      const auto &loc_slice = NameLoc::get(rewriter.getStringAttr(name_slice));
      auto new_slice = rewriter.create<tpu::SliceOp>(
          loc_slice, squeeze.getOutput().getType(), operands, attrs);
      squeeze.getOutput().replaceAllUsesWith(new_slice.getOutput());
      rewriter.eraseOp(squeeze);
      rewriter.eraseOp(slice);
      change = success();
    }
    if (std::distance(op->user_begin(), op->user_end()) == 0) {
      rewriter.eraseOp(op);
    }
    return change;
  }
};

// merge_mode = 1: only merge weight oc
// merge_mode = 2: merge weight ic and oc
static Value merge_conv_weight(PatternRewriter &rewriter, Operation *op,
                               Value w0, Value w1, int merge_mode,
                               std::string suffix) {
  auto op0 = w0.getDefiningOp();
  auto op1 = w1.getDefiningOp();
  auto weight0 = dyn_cast<top::WeightOp>(op0);
  auto weight1 = dyn_cast<top::WeightOp>(op1);
  auto data0 = weight0.read<int8_t>();
  auto data1 = weight1.read<int8_t>();

  auto wshape0 = module::getShape(weight0.getOutput());
  // auto wshape1 = module::getShape(weight1.getOutput());
  std::shared_ptr<std::vector<int8_t>> new_data;
  std::vector<int64_t> new_shape{wshape0[0], wshape0[1], wshape0[2],
                                 wshape0[3]};
  if (merge_mode == 1) {
    new_shape[0] = wshape0[0] * 2;
    int size = data0->size() + data1->size();
    new_data = std::make_shared<std::vector<int8_t>>(size, 0);
    std::copy(data0->data(), data0->data() + data0->size(), new_data->data());
    std::copy(data1->data(), data1->data() + data1->size(),
              new_data->data() + data0->size());
  } else if (merge_mode == 2) {
    new_shape[0] = wshape0[0] * 2;
    new_shape[1] = wshape0[1] * 2;
    int size = (data0->size() + data1->size()) * 2;
    new_data = std::make_shared<std::vector<int8_t>>(size, 0);
    int size0 = data0->size() * 2;
    int offset = wshape0[1] * wshape0[2] * wshape0[3];
    for (int i = 0; i < wshape0[0]; ++i) {
      std::copy(data0->data() + offset * i, data0->data() + offset * (i + 1),
                new_data->data() + offset * i * 2);
      std::copy(data1->data() + offset * i, data1->data() + offset * (i + 1),
                new_data->data() + size0 + offset * i * 2 + offset);
    }
  } else {
    return nullptr;
  }
  // auto stype = module::getStorageType(weight0.getOutput());
  rewriter.setInsertionPointAfter(op);
  auto new_type = RankedTensorType::get(
      new_shape, module::getElementType(weight0.getOutput()));
  return top::WeightOp::create<int8_t>(op0, suffix, *new_data, new_type);
}

static Value merge_conv_bias(PatternRewriter &rewriter, Operation *op, Value b0,
                             Value b1, std::string suffix) {
  auto op0 = b0.getDefiningOp();
  auto op1 = b1.getDefiningOp();
  auto bias0 = dyn_cast<top::WeightOp>(op0);
  auto bias1 = dyn_cast<top::WeightOp>(op1);
  auto data0 = bias0.read<int32_t>();
  auto data1 = bias1.read<int32_t>();

  auto bshape0 = module::getShape(bias0.getOutput());
  std::vector<int64_t> new_shape{bshape0[0], bshape0[1] * 2, bshape0[2],
                                 bshape0[3]};
  std::shared_ptr<std::vector<int32_t>> new_data =
      std::make_shared<std::vector<int32_t>>(new_shape[1], 0);
  std::copy(data0->data(), data0->data() + data0->size(), new_data->data());
  std::copy(data1->data(), data1->data() + data1->size(),
            new_data->data() + data0->size());
  // auto stype = module::getStorageType(bias0.getOutput());
  rewriter.setInsertionPointAfter(op);
  auto new_type = RankedTensorType::get(
      new_shape, module::getElementType(bias0.getOutput()));
  return top::WeightOp::create<int32_t>(op0, suffix, *new_data, new_type);
}

static tpu::SliceOp create_slice_op(PatternRewriter &rewriter, Operation *op,
                                    Value input, int offset, int end,
                                    std::string suffix) {
  auto name = module::getName(input);
  auto new_name = name.str() + "_" + suffix;
  auto new_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  rewriter.setInsertionPointAfterValue(input);
  std::vector<int64_t> new_shape = module::getShape(input);
  std::vector<int64_t> offset_v(new_shape.size(), 0);
  std::vector<int64_t> step_v(new_shape.size(), 1);
  std::vector<int64_t> end_v = new_shape;
  offset_v[1] = offset;
  end_v[1] = end;
  new_shape[1] = end - offset;
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset_v)));
  attrs.push_back(
      rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(step_v)));
  attrs.push_back(
      rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(end_v)));
  attrs.push_back(rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(1)));
  auto new_type = module::getTypeLike(input, new_shape);
  auto none = module::getNoneOp(input.getDefiningOp());
  auto new_op = rewriter.create<tpu::SliceOp>(
      new_loc, new_type, ValueRange{input, none, none, none, none}, attrs);

  return new_op;
}

// Conv merge requant
class ConvMergeRequant : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  using OpRewriterPatternEx::OpRewriterPatternEx;
  ConvMergeRequant(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "ConvMergeRequant",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::Conv2DOp op,
                                    PatternRewriter &rewriter) const override {
    auto left = op.getInput();
    auto right = op.getFilter();
    auto bias = op.getBias();

    auto stype = module::getStorageType(left);
    if (!stype.isInteger(8)) {
      return failure();
    }

    if (op->hasOneUse() == false) {
      return failure();
    }

    std::vector<int64_t> rshift_v;
    std::vector<int64_t> multiplier_v;
    auto shape = module::getShape(op.getOutput());
    if (auto requant_op =
            dyn_cast<tpu::RequantIntOp>(*(op.getOutput().getUsers().begin()))) {
      if (requant_op.getQuantMode() != tpu::RequantMode::MultiplierShift)
        return failure();
      // if (!module::getStorageType(requant_op.getOutput()).isInteger(8))
      //   return failure();
      // Conv merge requant_int
      multiplier_v.assign(shape[1], requant_op.getMultiplier());
      rshift_v.assign(shape[1], requant_op.getRshift());
      op.setMultiplierAttr(
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v}));
      op.setRshiftAttr(rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v}));
      op.setQuantModeAttr(requant_op.getQuantModeAttr());
      op.setRoundModeAttr(requant_op.getRoundModeAttr());
      rewriter.setInsertionPointAfter(op);
      auto s_op = rewriter.create<tpu::Conv2DOp>(
          requant_op->getLoc(), requant_op.getOutput().getType(),
          ValueRange{left, right, bias}, op->getAttrs());
      requant_op.replaceAllUsesWith(s_op.getOutput());
      return success();
    }

    if (auto requant_op = dyn_cast<tpu::RequantIntAxisOp>(
            *(op.getOutput().getUsers().begin()))) {
      if (requant_op.getQuantMode() != tpu::RequantMode::MultiplierShift)
        return failure();
      // if (!module::getStorageType(requant_op.getOutput()).isInteger(8))
      //   return failure();
      // Conv merge requant_int_axis
      auto requant =
          dyn_cast<top::WeightOp>(requant_op.getQuant().getDefiningOp());
      auto data = requant.read<int32_t>();
      if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
          module::isSGTPUV8()) {
        for (int i = 0; i < shape[1]; ++i) {
          multiplier_v.push_back(data->data()[i * 2]);
          rshift_v.push_back(-(data->data()[i * 2 + 1] & 0xffff));
        }
      } else {
        for (int i = 0; i < shape[1]; ++i) {
          multiplier_v.push_back(data->data()[i * 3]);
          rshift_v.push_back(-(data->data()[i * 3 + 1]));
        }
      }
      op.setMultiplierAttr(
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v}));
      op.setRshiftAttr(rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v}));
      op.setQuantModeAttr(requant_op.getQuantModeAttr());
      op.setRoundModeAttr(requant_op.getRoundModeAttr());
      rewriter.setInsertionPointAfter(op);
      auto s_op = rewriter.create<tpu::Conv2DOp>(
          requant_op->getLoc(), requant_op.getOutput().getType(),
          ValueRange{left, right, bias}, op->getAttrs());
      requant_op.replaceAllUsesWith(s_op.getOutput());
      return success();
    }

    return failure();
  }
};

class ConvMergePattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  ConvMergePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "ConvMergePattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::Conv2DOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (!module::isUniformQuantized(in)) {
      return failure();
    }
    auto pre_op = in.getDefiningOp();

    auto ins = pre_op->getOperands();
    if (!isa<tpu::ConcatOp>(pre_op) || ins.size() != 3) {
      return failure();
    }

    tpu::Conv2DOp conv1 = nullptr, conv2 = nullptr, conv3 = nullptr;
    for (size_t i = 0; i < ins.size(); ++i) {
      if (!isa<tpu::Conv2DOp>(ins[i].getDefiningOp())) {
        return failure();
      }

      auto op_ = ins[i].getDefiningOp();
      int num_conv = 1;
      int num_users = std::distance(op_->user_begin(), op_->user_end());
      while (num_users == 1) {
        op_ = op_->getOperand(0).getDefiningOp();
        num_users = std::distance(op_->user_begin(), op_->user_end());
        if (num_users == 1) {
          if (!isa<tpu::Conv2DOp>(op_)) {
            return failure();
          }
          num_conv++;
        }
      }
      if (num_conv == 1) {
        conv1 = cast<tpu::Conv2DOp>(ins[i].getDefiningOp());
      } else if (num_conv == 2) {
        conv2 = cast<tpu::Conv2DOp>(ins[i].getDefiningOp());
      } else if (num_conv == 3) {
        conv3 = cast<tpu::Conv2DOp>(ins[i].getDefiningOp());
      } else {
        return failure();
      }
    }
    if (conv1 == nullptr || conv2 == nullptr || conv3 == nullptr) {
      return failure();
    }
    auto conv2_ishape = module::getShape(conv2->getOperand(0));
    auto conv2_wshape = module::getShape(conv2->getOperand(1));
    auto conv3_ishape = module::getShape(conv3->getOperand(0));
    auto conv3_wshape = module::getShape(conv3->getOperand(1));
    if (conv2_ishape[1] != 32 || conv2_wshape[0] != 32 ||
        conv3_ishape[1] != 32 || conv3_wshape[0] != 32) {
      return failure();
    }
    auto conv4 = dyn_cast<tpu::Conv2DOp>(conv2->getOperand(0).getDefiningOp());
    auto conv5 = dyn_cast<tpu::Conv2DOp>(conv3->getOperand(0).getDefiningOp());
    auto conv4_ishape = module::getShape(conv4->getOperand(0));
    auto conv4_wshape = module::getShape(conv4->getOperand(1));
    auto conv5_ishape = module::getShape(conv5->getOperand(0));
    auto conv5_wshape = module::getShape(conv5->getOperand(1));
    if (conv4_ishape[1] != 256 || conv4_wshape[0] != 32 ||
        conv5_ishape[1] != 32 || conv5_wshape[0] != 32) {
      return failure();
    }
    auto conv6 = dyn_cast<tpu::Conv2DOp>(conv5->getOperand(0).getDefiningOp());
    auto conv6_ishape = module::getShape(conv6->getOperand(0));
    auto conv6_wshape = module::getShape(conv6->getOperand(1));
    if (conv6_ishape[1] != 256 || conv6_wshape[0] != 32) {
      return failure();
    }
    auto src_op = conv6.getInput().getDefiningOp();

    // merge conv4 and conv6
    auto new_weight0 = merge_conv_weight(rewriter, src_op, conv4->getOperand(1),
                                         conv6->getOperand(1), 1, "merge_0");
    auto new_bias0 =
        merge_conv_bias(rewriter, new_weight0.getDefiningOp(),
                        conv4->getOperand(2), conv6->getOperand(2), "merge_0");
    auto multi4 =
        module::getI64Array(conv4.getMultiplier(), conv4_wshape[0], 1);
    auto multi6 =
        module::getI64Array(conv6.getMultiplier(), conv6_wshape[0], 1);
    auto rshift4 = module::getI64Array(conv4.getRshift(), conv4_wshape[0], 0);
    auto rshift6 = module::getI64Array(conv6.getRshift(), conv6_wshape[0], 0);
    std::vector<int64_t> new_multi0(conv4_wshape[0] * 2, 0);
    std::vector<int64_t> new_rshift0(conv4_wshape[0] * 2, 0);
    std::copy(multi4->begin(), multi4->end(), new_multi0.begin());
    std::copy(rshift4->begin(), rshift4->end(), new_rshift0.begin());
    std::copy(multi6->begin(), multi6->end(),
              new_multi0.begin() + conv4_wshape[0]);
    std::copy(rshift6->begin(), rshift6->end(),
              new_rshift0.begin() + conv4_wshape[0]);

    std::vector<int64_t> conv6_oshape = module::getShape(conv6.getOutput());
    conv6_oshape[1] = conv4_wshape[0] + conv6_wshape[0];
    std::string conv_name6 =
        module::getName(conv6.getOperation()).str() + "_merge_0";
    auto new_loc0 = NameLoc::get(rewriter.getStringAttr(conv_name6));
    auto new_type0 = module::getTypeLike(conv6.getOutput(), conv6_oshape);
    std::vector<Value> operands0{src_op->getResult(0), new_weight0, new_bias0};
    std::vector<NamedAttribute> attrs0;
    for (auto &attr : conv6->getAttrs()) {
      attrs0.push_back(attr);
    }
    rewriter.setInsertionPointAfter(src_op);
    auto new_conv0 =
        rewriter.create<tpu::Conv2DOp>(new_loc0, new_type0, operands0, attrs0);
    new_conv0.setMultiplierAttr(rewriter.getI64ArrayAttr(new_multi0));
    new_conv0.setRshiftAttr(rewriter.getI64ArrayAttr(new_rshift0));
    new_weight0.getDefiningOp()->moveBefore(new_conv0);
    new_bias0.getDefiningOp()->moveBefore(new_conv0);

    // merge conv2 and conv5
    auto new_weight1 =
        merge_conv_weight(rewriter, new_conv0, conv2->getOperand(1),
                          conv5->getOperand(1), 2, "merge_1");
    auto new_bias1 =
        merge_conv_bias(rewriter, new_weight1.getDefiningOp(),
                        conv2->getOperand(2), conv5->getOperand(2), "merge_1");
    auto multi2 =
        module::getI64Array(conv2.getMultiplier(), conv2_wshape[0], 1);
    auto multi5 =
        module::getI64Array(conv5.getMultiplier(), conv5_wshape[0], 1);
    auto rshift2 = module::getI64Array(conv2.getRshift(), conv2_wshape[0], 0);
    auto rshift5 = module::getI64Array(conv5.getRshift(), conv5_wshape[0], 0);
    std::vector<int64_t> new_multi1(conv2_wshape[0] * 2, 0);
    std::vector<int64_t> new_rshift1(conv5_wshape[0] * 2, 0);
    std::copy(multi2->begin(), multi2->end(), new_multi1.begin());
    std::copy(rshift2->begin(), rshift2->end(), new_rshift1.begin());
    std::copy(multi5->begin(), multi5->end(),
              new_multi1.begin() + conv5_wshape[0]);
    std::copy(rshift5->begin(), rshift5->end(),
              new_rshift1.begin() + conv5_wshape[0]);
    std::vector<int64_t> conv5_oshape = module::getShape(conv5.getOutput());
    conv5_oshape[1] = conv2_wshape[0] + conv5_wshape[0];
    std::string conv_name5 =
        module::getName(conv5.getOperation()).str() + "_merge_1";
    auto new_loc1 = NameLoc::get(rewriter.getStringAttr(conv_name5));
    auto new_type1 = module::getTypeLike(conv5.getOutput(), conv5_oshape);
    std::vector<Value> operands1{new_conv0.getResult(), new_weight1, new_bias1};
    std::vector<NamedAttribute> attrs1;
    for (auto &attr : conv2->getAttrs()) {
      attrs1.push_back(attr);
    }
    rewriter.setInsertionPointAfter(new_conv0);
    auto new_conv1 =
        rewriter.create<tpu::Conv2DOp>(new_loc1, new_type1, operands1, attrs1);
    new_conv1.setMultiplierAttr(rewriter.getI64ArrayAttr(new_multi1));
    new_conv1.setRshiftAttr(rewriter.getI64ArrayAttr(new_rshift1));
    new_weight1.getDefiningOp()->moveBefore(new_conv1);
    new_bias1.getDefiningOp()->moveBefore(new_conv1);

    // create SliceOp
    auto slice0 = create_slice_op(rewriter, new_conv1, new_conv1.getResult(), 0,
                                  32, "slice_0");
    auto slice1 = create_slice_op(rewriter, slice0, new_conv1.getResult(), 32,
                                  64, "slice_1");

    // replace
    conv2.replaceAllUsesWith(slice0.getResult());
    conv5.replaceAllUsesWith(slice1.getResult());
    rewriter.eraseOp(conv5);
    rewriter.eraseOp(conv2);
    rewriter.eraseOp(conv6);
    rewriter.eraseOp(conv4);
    return success();
  }
};

class NoneZeroFixRowMajor : public OpRewriterPatternEx<tpu::NonZeroOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  NoneZeroFixRowMajor(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::NonZeroOp>(context, "NoneZeroFixRowMajor",
                                            benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::NonZeroOp op,
                                    PatternRewriter &rewriter) const override {
    const int order = op.getOrder().str() == "ColMajor" ? 0 : 1;
    if (order == 0) {
      return failure();
    }
    auto type = op.getResult().getType();
    op->setAttr("order", rewriter.getStringAttr("ColMajor"));
    auto out_shape = module::getShape(op.getOutput());
    std::vector<int64_t> new_out_shape = {out_shape[1], out_shape[0]};
    module::setShape(op.getOutput(), new_out_shape);

    rewriter.setInsertionPointAfter(op);
    auto permute_out = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getOutput()).str() + "_permute"));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> Porder = {1, 0};
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(Porder)));
    auto permute_op = rewriter.create<tpu::PermuteOp>(
        permute_out, type, ValueRange{op.getOutput(), module::getNoneOp(op)},
        attrs);
    rewriter.replaceAllUsesExcept(op.getOutput(), permute_op.getOutput(),
                                  permute_op);
    return success();
  }
};

class SplitReduceL2Pattern : public OpRewriterPatternEx<tpu::ReduceOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  SplitReduceL2Pattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ReduceOp>(context, "SplitReduceL2Pattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    /* ReduceL2(1x6x8x65536)->Reshape(1x48x256x256)+ReduceL2(1x48x256x1)+ReduceL2(1x48x1x1)+Reshape(1x6x8x1)
     * ReduceL2(1x48x65536)->Reshape(48x256x256)+ReduceL2(48x256x1)+ReduceL2(48x1x1)+Reshape(1x48x1)
     */
    // TODO : support quant type; consider the divisor of the reduced dim;
    auto mode = op.getMode();
    auto input = op.getInput();
    auto output = op.getOutput();
    auto input_shape = module::getShape(input);
    int input_dim = input_shape.size();
    // if (input_shape.size() != 3) {
    //   return failure();
    // }
    auto axes = module::getI64Array(op.getAxes());
    if (axes->size() != 1)
      return failure();
    int split_dim = axes->at(0);
    if (input_dim < 3 || split_dim != (input_dim - 1) ||
        input_shape[split_dim] != 65536)
      return failure();

    auto name = module::getName(input);
    std::vector<Value> operands;
    std::vector<int64_t> reshape0_shape = input_shape;
    reshape0_shape[split_dim - 2] *= reshape0_shape[split_dim - 1];
    reshape0_shape[split_dim - 1] = reshape0_shape[split_dim] = 256;
    auto reshape0_type = module::getTypeLike(output, reshape0_shape);
    auto loc_reshape0 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape0"));
    rewriter.setInsertionPointAfterValue(input);
    auto reshape_op0 = rewriter.create<tpu::ReshapeOp>(
        loc_reshape0, reshape0_type, ValueRange{input});

    std::vector<int64_t> reducel2_0_shape = reshape0_shape;
    reducel2_0_shape[split_dim] = 1;
    auto reducel2_type0 = module::getTypeLike(output, reducel2_0_shape);
    auto loc_reduce0 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reduce0"));
    rewriter.setInsertionPointAfter(reshape_op0);
    operands.push_back(reshape_op0.getOutput());
    auto noneOp = module::getNoneOp(op);
    for (int i = operands.size(); i < 3; i++) {
      operands.push_back(noneOp);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr(mode)));
    attrs.push_back(rewriter.getNamedAttr("axes", op.getAxes()));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reducel2_op0 = rewriter.create<tpu::ReduceOp>(
        loc_reduce0, reducel2_type0, operands, attrs);

    operands.clear();
    attrs.clear();
    std::vector<int64_t> reducel2_1_shape = reducel2_0_shape;
    reducel2_1_shape[split_dim - 1] = 1;
    auto reducel2_type1 = module::getTypeLike(output, reducel2_1_shape);
    auto loc_reduce1 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reduce1"));
    rewriter.setInsertionPointAfter(reducel2_op0);
    operands.push_back(reducel2_op0.getOutput());
    for (int i = operands.size(); i < 3; i++) {
      operands.push_back(noneOp);
    }
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr(mode)));
    // op.setAxesAttr(rewriter.getI64ArrayAttr({axes->at(0) - 1}));
    attrs.push_back(rewriter.getNamedAttr(
        "axes", rewriter.getI64ArrayAttr({axes->at(0) - 1})));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reducel2_op1 = rewriter.create<tpu::ReduceOp>(
        loc_reduce1, reducel2_type1, operands, attrs);

    std::vector<int64_t> reshape1_shape = input_shape;
    reshape1_shape[split_dim] = 1;
    auto reshape1_type = module::getTypeLike(output, reshape1_shape);
    auto loc_reshape1 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape1"));
    rewriter.setInsertionPointAfter(reducel2_op1);
    auto reshape_op1 = rewriter.create<tpu::ReshapeOp>(
        loc_reshape1, reshape1_type, ValueRange{reducel2_op1.getOutput()});
    rewriter.replaceAllUsesWith(op.getOutput(), reshape_op1.getOutput());

    rewriter.eraseOp(op);
    return success();
  }
};

class Reduce2AxesPattern : public OpRewriterPatternEx<tpu::ReduceOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  Reduce2AxesPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::ReduceOp>(context, "Reduce2AxesPattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    /* ReduceL2(1x4x256x256,axes[2,3],keep_dims=false)->ReduceL2(1x4x256x256)+ReduceL2(1x4x256)
     */
    if (!(module::isBM1688() || module::isSG2380() || module::isMARS3() ||
          module::isSGTPUV8())) {
      return failure();
    }
    auto mode = op.getMode();
    auto input = op.getInput();
    auto output = op.getOutput();
    auto input_shape = module::getShape(input);
    int input_dim = input_shape.size();
    auto axes = module::getI64Array(op.getAxes());
    if (op.getKeepdims() || axes->size() != 2 || input_dim < 3 ||
        axes->at(0) != input_dim - 2 || axes->at(1) != input_dim - 1 ||
        input_shape[axes->at(0)] * input_shape[axes->at(1)] < 65536)
      return failure();

    auto name = module::getName(input);
    std::vector<Value> operands;

    std::vector<int64_t> reducel2_0_shape = input_shape;
    if (!reducel2_0_shape.empty()) {
      reducel2_0_shape.resize(reducel2_0_shape.size() - 1);
    }
    auto reducel2_type0 = module::getTypeLike(output, reducel2_0_shape);
    auto loc_reduce0 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reduce0"));
    // rewriter.setInsertionPointAfterValue(input);
    operands.push_back(input);
    auto noneOp = module::getNoneOp(op);
    for (int i = operands.size(); i < 3; i++) {
      operands.push_back(noneOp);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr(mode)));
    attrs.push_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({axes->at(1)})));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reducel2_op0 = rewriter.create<tpu::ReduceOp>(
        loc_reduce0, reducel2_type0, operands, attrs);

    operands.clear();
    attrs.clear();
    std::vector<int64_t> reducel2_1_shape = reducel2_0_shape;
    if (!reducel2_1_shape.empty()) {
      reducel2_1_shape.resize(reducel2_1_shape.size() - 1);
    }
    auto reducel2_type1 = module::getTypeLike(output, reducel2_1_shape);
    auto loc_reduce1 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reduce1"));
    // rewriter.setInsertionPointAfterValue(reducel2_op0.getOutput());
    operands.push_back(reducel2_op0.getOutput());
    for (int i = operands.size(); i < 3; i++) {
      operands.push_back(noneOp);
    }
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr(mode)));
    attrs.push_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({axes->at(0)})));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reducel2_op1 = rewriter.create<tpu::ReduceOp>(
        loc_reduce1, reducel2_type1, operands, attrs);

    rewriter.replaceAllUsesWith(op.getOutput(), reducel2_op1.getOutput());

    rewriter.eraseOp(op);
    return success();
  }
};

/**
 *                 Slice
 *  \                   \
 *    MatMul   ->         MatMul => Add
 *  /                   /
 *                 Slice
 */
class SplitMatmulPattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  SplitMatmulPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "SplitMatmulPattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    /*
     * MatMul(1x1x48x65536,1x1x65536x48) -> Slice =>
     * MatMul(1x1x48x16384,1x1x16384x48) => Add
     */
    /*
     * MatMul(1x48x65536,1x65536x48) -> Slice =>
     * MatMul(1x48x16384,1x16384x48) => Add
     */
    // TODO : judge whether K is splited; what if other shape > 65535
    // if (!module::isBM1688()) {
    //   return failure();
    // }
    auto left = op.getInput();
    auto right = op.getRight();
    auto left_shape = module::getShape(left);
    auto right_shape = module::getShape(right);
    auto output_shape = module::getShape(op.getOutput());
    int left_dim = left_shape.size();
    int right_dim = right_shape.size();
    auto l_trans = op.getLeftTranspose();
    auto r_trans = op.getRightTranspose();

    if ((left_dim != 4 && left_dim != 3) || left_dim != right_dim ||
        left_shape[left_dim - 1] < 16384 || l_trans || !r_trans) {
      return failure();
    }
    int left_K_size = left_shape[left_dim - 1];
    std::vector<int64_t> left_offset(left_dim, 0);
    std::vector<int64_t> left_steps(left_dim, 1);
    std::vector<int64_t> left_ends(left_dim, -1);
    std::vector<int64_t> right_offset(right_dim, 0);
    std::vector<int64_t> right_steps(right_dim, 1);
    std::vector<int64_t> right_ends(right_dim, -1);
    auto left_name = module::getName(left);
    auto right_name = module::getName(right);

    auto name = module::getName(op.getOutput());
    std::vector<Value> operands;

    int secs = 8;
    if (left_shape[left_dim - 1] % 8 != 0)
      return failure();
    std::vector<int64_t> new_left_shapes = left_shape;
    new_left_shapes[left_dim - 1] /= secs;
    std::vector<int64_t> new_right_shapes = right_shape;
    new_right_shapes[right_dim - 1] /= secs;
    std::vector<NamedAttribute> attrs;
    for (int i = 0; i < secs; i++) {
      left_offset[left_dim - 1] = left_K_size * i / secs;
      left_ends[left_dim - 1] = left_K_size * (i + 1) / secs;
      attrs.push_back(rewriter.getNamedAttr(
          "axes", rewriter.getI64ArrayAttr(left_dim - 1)));
      attrs.push_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr(left_offset)));
      attrs.push_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(left_steps)));
      attrs.push_back(
          rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(left_ends)));

      auto loc = NameLoc::get(rewriter.getStringAttr(
          left_name.str() + "_slice" + std::to_string(i)));
      auto left_type = module::getTypeLike(left, new_left_shapes);
      rewriter.setInsertionPointAfterValue(left);
      auto none = module::getNoneOp(left.getDefiningOp());
      auto left_op = rewriter.create<tpu::SliceOp>(
          loc, left_type, ValueRange{left, none, none, none, none}, attrs);

      attrs.clear();
      right_offset[right_dim - 1] = left_K_size * i / secs;
      right_ends[right_dim - 1] = left_K_size * (i + 1) / secs;
      attrs.push_back(rewriter.getNamedAttr(
          "axes", rewriter.getI64ArrayAttr(right_dim - 1)));
      attrs.push_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr(right_offset)));
      attrs.push_back(rewriter.getNamedAttr(
          "steps", rewriter.getI64ArrayAttr(right_steps)));
      attrs.push_back(
          rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(right_ends)));
      loc = NameLoc::get(rewriter.getStringAttr(right_name.str() + "_slice" +
                                                std::to_string(i)));
      auto right_type = module::getTypeLike(right, new_right_shapes);
      rewriter.setInsertionPointAfterValue(right);
      auto right_op = rewriter.create<tpu::SliceOp>(
          loc, right_type, ValueRange{right, none, none, none, none}, attrs);

      attrs.clear();
      rewriter.setInsertionPointAfter(op);
      auto new_matmul_op = rewriter.clone(*op);
      module::setLocSuffix(new_matmul_op, std::to_string(i));
      new_matmul_op->setOperand(0, left_op->getResult(0));
      new_matmul_op->setOperand(1, right_op->getResult(0));
      module::setShape(new_matmul_op->getResult(0), output_shape);

      operands.push_back(new_matmul_op->getResult(0));
    }
    // Value insertpoint = operands[0];
    // if(operands[1].getDefiningOp()->getLoc() >
    // operands[0].getDefiningOp()->getLoc())
    //   insertpoint = operands[1];
    rewriter.setInsertionPointAfterValue(operands[0]);
    auto loc = NameLoc::get(
        rewriter.getStringAttr(name.str() + "_add" + std::to_string(0)));
    auto add_op = rewriter.create<tpu::AddOp>(
        loc, operands[0].getType(), mlir::ValueRange{operands[0], operands[1]});
    for (int i = 1; i < operands.size() - 1; i++) {
      // if(operands[i+1].getDefiningOp()->getLoc() > add_op->getLoc())
      //   insertpoint = operands[1];
      rewriter.setInsertionPointAfterValue(add_op);
      loc = NameLoc::get(
          rewriter.getStringAttr(name.str() + "_add" + std::to_string(i)));
      add_op = rewriter.create<tpu::AddOp>(
          loc, operands[i].getType(),
          mlir::ValueRange{add_op.getOutput(), operands[i + 1]});
    }

    rewriter.replaceAllUsesWith(op.getOutput(), add_op.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

struct GridSamplerFusePattern : public OpRewriterPatternEx<tpu::GridSamplerOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  GridSamplerFusePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::GridSamplerOp>(
            context, "GridSamplerFusePattern", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::GridSamplerOp op,
                                    PatternRewriter &rewriter) const override {
    auto dims = module::getShape(op.getInput()).size();
    if (dims != 5) {
      return failure();
    }
    auto grid = op.getGrid();
    auto pre_op = grid.getDefiningOp();
    if (!pre_op->hasOneUse()) {
      return failure();
    }
    if (isa<tpu::PermuteOp>(pre_op)) {
      auto permute = cast<tpu::PermuteOp>(pre_op);
      auto order = module::getI64Array(permute.getOrder());
      std::vector<int64_t> ps = {0, 2, 3, 4, 1};
      if (*order != ps) {
        return failure();
      }
      op->setAttr("need_permute", rewriter.getBoolAttr(true));
      op->setOperand(1, permute.getInput());
      rewriter.eraseOp(permute);
    } else if (isa<tpu::MulConstOp>(pre_op)) {
      auto scale = op.getScale().convertToDouble();
      auto mean = op.getMean().convertToDouble();
      if (scale != 1.0 || mean != 0.0) {
        return failure();
      }
      auto mulconst = cast<tpu::MulConstOp>(pre_op);
      auto const_val = mulconst.getConstVal().convertToDouble();
      op->setAttr("scale", rewriter.getF64FloatAttr(const_val));
      op->setOperand(1, mulconst.getInput());
      rewriter.eraseOp(mulconst);
    } else if (isa<tpu::AddConstOp>(pre_op)) {
      auto mean = op.getMean().convertToDouble();
      if (mean != 0.0) {
        return failure();
      }
      auto addconst = cast<tpu::AddConstOp>(pre_op);
      auto const_val = addconst.getConstVal().convertToDouble();
      op->setAttr("mean", rewriter.getF64FloatAttr(const_val));
      op->setOperand(1, addconst.getInput());
      rewriter.eraseOp(addconst);
    } else {
      return failure();
    }
    return success();
  }
};

// Some special gridsampler with a large batch size and H = 1
//
// input_data -> permute            ->
//         mulconst                    gridsample -> permute
// grid -> addconst (Normalization) ->
//         concat
struct CanCutGridSamplerFusePattern
    : public OpRewriterPatternEx<tpu::GridSamplerOp> {

  CanCutGridSamplerFusePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::GridSamplerOp>(
            context, "CanCutGridSamplerFusePattern", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::GridSamplerOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op.getOutput().hasOneUse()) {
      return failure();
    }
    auto before_op = op.getInput().getDefiningOp();
    auto next_op = *op.getOutput().user_begin();
    if (!isa<tpu::PermuteOp>(before_op) && !isa<tpu::PermuteOp>(next_op)) {
      return failure();
    }
    auto grid = op.getGrid();
    auto concat_op = grid.getDefiningOp<tpu::ConcatOp>();
    if (concat_op.getInputs().size() != 2) {
      return failure();
    }
    bool is_weight = false;
    Value concat_input;

    for (auto input : concat_op.getInputs()) {
      if (isa<top::WeightOp>(input.getDefiningOp())) {
        is_weight = true;
      } else {
        concat_input = input;
      }
    }
    if (!is_weight) {
      return failure();
    }
    auto addconst = concat_input.getDefiningOp<tpu::AddConstOp>();
    if (!addconst) {
      return failure();
    }
    auto mulconst = addconst.getInput().getDefiningOp<tpu::MulConstOp>();
    if (!mulconst) {
      return failure();
    }
    auto permute = mulconst.getInput().getDefiningOp<tpu::PermuteOp>();
    if (!permute) {
      return failure();
    }

    if (!next_op->getResult(0).use_empty()) {
      auto next_next_op = *next_op->getResult(0).user_begin();
      next_next_op->setOperand(0, op.getOutput());
    }
    op->setOperand(0, before_op->getOperand(0));
    op->setOperand(1, permute.getInput());
    op->getResult(0).setType(next_op->getResult(0).getType());
    op->setAttr("need_permute", rewriter.getBoolAttr(true));
    rewriter.replaceOp(before_op, before_op->getOperand(0));
    rewriter.replaceOp(next_op, ArrayRef<Value>{op.getResult()});

    rewriter.eraseOp(concat_op);
    rewriter.eraseOp(addconst);
    rewriter.eraseOp(mulconst);
    rewriter.eraseOp(permute);
    return success();
  }
};

// MatMul  +  RequantIntAxis ->  MatMul
class MatMulRequantIntFusion : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  MatMulRequantIntFusion(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "MatMulRequantIntFusion",
                                           benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    bool m_fuse_rq = op.getFuseRq();
    if (m_fuse_rq) // If matmul supports per-channel quantization in the
                   // lowering stage, don't fuse rq!
      failure();

    if (!(module::isBM1684X() || module::isBM1688()) || !op->hasOneUse()) {
      return failure();
    }

    auto nextOp = *op.getOutput().user_begin();
    if (!isa<tpu::RequantIntAxisOp>(nextOp))
      return failure();

    auto requantIntAxisOp = dyn_cast<tpu::RequantIntAxisOp>(nextOp);
    bool fuse_rq_axis = requantIntAxisOp.getFuseRqAxis();
    if (!fuse_rq_axis)
      return failure();

    auto shape = module::getShape(requantIntAxisOp.getOutput());
    auto quantValueOp =
        requantIntAxisOp.getQuant().getDefiningOp<top::WeightOp>();
    auto quantDataPtr = quantValueOp.read<int32_t>();
    auto &quantData = *quantDataPtr;

    int channels = shape[shape.size() - 1];
    std::vector<int64_t> new_multi0(channels);
    std::vector<int64_t> new_rshift0(1);

    for (int i = 0; i < channels; ++i) {
      if (module::isBM1684X()) {
        new_multi0[i] = quantData[i * 3]; // multi
      } else if (module::isBM1688()) {
        new_multi0[i] = quantData[i * 2]; // multi
      }
    }
    new_rshift0[0] = quantData[1];

    rewriter.setInsertionPoint(op);
    std::vector<int32_t> reshaped_multi0(channels);
    std::copy(new_multi0.begin(), new_multi0.end(), reshaped_multi0.begin());
    std::vector<int64_t> multi_shape = {1, channels};
    multi_shape.insert(multi_shape.begin(), shape.size() - multi_shape.size(),
                       1);
    auto multi_type = mlir::RankedTensorType::get(
        multi_shape, rewriter.getIntegerType(32, true));
    auto weight_op =
        top::WeightOp::create(op, "i32", reshaped_multi0, multi_type);

    std::vector<mlir::Value> operands(op.getOperands().begin(),
                                      op.getOperands().end());
    operands[operands.size() - 2] = weight_op;

    auto newMatmulOp = rewriter.create<tpu::MatMulOp>(
        requantIntAxisOp->getLoc(), requantIntAxisOp.getOutput().getType(),
        operands, op->getAttrs());

    // newMatmulOp.setMultipliersAttr(rewriter.getI64ArrayAttr(new_multi0));
    newMatmulOp.setRshiftsAttr(rewriter.getI64ArrayAttr(new_rshift0));
    newMatmulOp.setFuseRqAttr(rewriter.getBoolAttr(true));
    auto round_mode = requantIntAxisOp.getRoundModeAttr().getValue();
    newMatmulOp.setRoundMode(round_mode);
    rewriter.replaceOp(requantIntAxisOp, newMatmulOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

class SplitQuantizedMLP2Pattern : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  SplitQuantizedMLP2Pattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::MatMulOp>(context, "SplitQuantizedMLP2Pattern",
                                           benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp matMulOp,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1684X() && !module::isBM1688()) {
      return failure();
    }
    bool f_fuse_rq = matMulOp.getFuseRq();
    if (!f_fuse_rq)
      return failure();
    if (!isa<top::WeightOp>(matMulOp.getRight().getDefiningOp())) {
      return failure();
    }
    auto lut_op = dyn_cast<tpu::LutOp>(matMulOp.getOperand(0).getDefiningOp());
    if (!lut_op) {
      return failure();
    }
    auto prevMatMulOp =
        dyn_cast<tpu::MatMulOp>(lut_op->getOperand(0).getDefiningOp());
    if (!prevMatMulOp) {
      return failure();
    }
    if (!isa<top::WeightOp>(prevMatMulOp.getRight().getDefiningOp())) {
      return failure();
    }
    bool l_fuse_rq = prevMatMulOp.getFuseRq();
    if (!l_fuse_rq)
      return failure();
    auto in_size = module::getNumElements(prevMatMulOp.getInput());
    auto out_size = module::getNumElements(matMulOp.getOutput()) *
                    4; // stored as INT32 before sumed together and requantized.
    auto w_shape = module::getShape(matMulOp.getRight());
    auto dim = w_shape.size();
    auto w_size = module::getNumElements(matMulOp.getRight());
    auto ceilDiv = [](int64_t a, int64_t b) -> int64_t {
      return (a + b - 1) / b;
    };
    // at tpu.Add(BinaryShift) timestep, min_Lmem = 3 x outsize + insize.
    if (BM168x::LMEM_BANKS <
        ceilDiv(3 * out_size / BM168x::NPU_NUM, BM168x::LMEM_BANK_BYTES) +
            ceilDiv(in_size / BM168x::NPU_NUM, BM168x::LMEM_BANK_BYTES)) {
      return failure();
    }
    // get split number
    // at tpu.matmul timestep, min_Lmem = 2 x outsize + insize + bias,lut size +
    // 2 * weightsize
    auto max_weight_banks =
        BM168x::LMEM_BANKS -
        ceilDiv(in_size / BM168x::NPU_NUM, BM168x::LMEM_BANK_BYTES) -
        ceilDiv(2 * out_size / BM168x::NPU_NUM, BM168x::LMEM_BANK_BYTES) - 1;
    auto max_weight_size = (int64_t)(max_weight_banks / 2) *
                           BM168x::LMEM_BANK_BYTES * BM168x::NPU_NUM;
    int split_num = 1;
    while (w_size > max_weight_size) {
      split_num *= 2;
      w_size /= 2;
      if (w_shape[dim - 1] / split_num < BM168x::NPU_NUM ||
          w_shape[dim - 1] % split_num != 0) {
        return failure();
      }
    }
    if (split_num == 1) {
      return failure();
    }

    rewriter.setInsertionPointAfter(matMulOp);
    auto nativeVar_0 = tpu_mlir::tpu::createSplitQuantizedMLP2(
        rewriter, prevMatMulOp, prevMatMulOp->getOperand(0), split_num);
    rewriter.replaceOp(matMulOp, nativeVar_0);
    return success();
  }
};

class SplitMixedQuantizedMLPPattern
    : public OpRewriterPatternEx<tpu::MatMulOp> {
public:
  SplitMixedQuantizedMLPPattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::MatMulOp>(
            context, "SplitMixedQuantizedMLPPattern", benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::MatMulOp matMulOp,
                                    PatternRewriter &rewriter) const override {
    bool f_fuse_rq = matMulOp.getFuseRq();
    if (f_fuse_rq)
      return failure();
    auto castOp = matMulOp->getOperand(0).getDefiningOp();
    if (!castOp || !isa<tpu::CastOp>(castOp)) {
      return failure();
    }

    auto lutOp = castOp->getOperand(0).getDefiningOp();
    if (!lutOp || !isa<tpu::LutOp>(lutOp)) {
      return failure();
    }

    auto prevMatMulOp = castOp->getOperand(0).getDefiningOp();
    if (!prevMatMulOp || !isa<tpu::MatMulOp>(prevMatMulOp)) {
      return failure();
    }
    auto prevMatMulOp_ = dyn_cast<tpu::MatMulOp>(prevMatMulOp);
    bool l_fuse_rq = prevMatMulOp_.getFuseRq();
    if (!prevMatMulOp->getOperand(0).hasOneUse() || l_fuse_rq) {
      return failure();
    }

    if (!isa<quant::UniformQuantizedType>(prevMatMulOp->getOperand(0)
                                              .getType()
                                              .cast<mlir::ShapedType>()
                                              .getElementType())) {
      return failure();
    }

    if (!isa<quant::CalibratedQuantizedType>(matMulOp->getResult(0)
                                                 .getType()
                                                 .cast<mlir::ShapedType>()
                                                 .getElementType())) {
      return failure();
    }

    if (!isa<top::WeightOp>(prevMatMulOp->getOperand(1).getDefiningOp())) {
      return failure();
    }

    auto nativeVar_0 = tpu_mlir::tpu::createSplitQuantizedMLP(
        rewriter, prevMatMulOp, prevMatMulOp->getOperand(0));

    rewriter.replaceOp(matMulOp, nativeVar_0);
    return success();
  }
};
class CastGradWeight : public OpRewriterPatternEx<tpu::ConvbwdOp> {
public:
  CastGradWeight(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::ConvbwdOp>(context, "CastGradWeight",
                                            benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::ConvbwdOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1690Family())
      return failure();
    auto attr = op.parseParam();
    auto grad_weight_enable = op.getGradWeightEnable();
    if (!grad_weight_enable) {
      return failure();
    }
    auto target_type = op.getResult(1).getType().cast<RankedTensorType>();
    auto target_shape = target_type.getShape();
    if (target_shape[2] != attr.kw && target_shape[3] != attr.kw) {
      return failure();
    }
    auto input_type = op.getOperand(0).getType().cast<RankedTensorType>();
    auto input_etype = input_type.getElementType();
    rewriter.setInsertionPointAfter(op);
    std::vector<Value> operands;
    for (auto &&in : op.getOperands())
      operands.emplace_back(in);
    std::vector<Type> new_types;
    std::vector<int64_t> gradweight_shape;
    int n = op->getNumResults();
    for (int i = 0; i < n; i++) {
      if (i == 1) {
        if (input_etype.isF32()) {
          gradweight_shape = {1, attr.oc, attr.ic * attr.kh * attr.kw, 1};
        } else if (input_etype.isF16()) {
          const int IC_PARALLEL = BM168x::ic_num(2);
          int64_t dw_h = ceiling_func(attr.ic, IC_PARALLEL) * attr.kh * attr.kw;
          gradweight_shape = {1, attr.oc, dw_h, IC_PARALLEL};
        }
        auto f16_type =
            RankedTensorType::get(gradweight_shape, rewriter.getF16Type());
        new_types.push_back(f16_type);
      } else {
        auto out = op.getResult(i);
        new_types.push_back(out.getType());
      }
    }
    auto module_fp16 = module::getMode() == module::Mode::F16;
    auto new_convbwd_op = rewriter.create<tpu::ConvbwdOp>(
        op.getLoc(), new_types, operands, op->getAttrs());
    if (module_fp16) {
      // auto op_name = module::getName(op.getResult(1));
      // auto cast_loc = NameLoc::get(
      //     rewriter.getStringAttr(op_name.str() + "cast_grad_weight"));
      // auto cast_type =
      //     RankedTensorType::get(gradweight_shape, rewriter.getF16Type());
      // auto cast_op = rewriter.create<tpu::CastOp>(
      //     cast_loc, cast_type, ValueRange{new_convbwd_op.getResult(1)});
      rewriter.replaceAllUsesWith(op.getResult(0), new_convbwd_op.getResult(0));
      rewriter.replaceAllUsesWith(op.getResult(1), new_convbwd_op.getResult(1));
      rewriter.replaceAllUsesWith(op.getResult(2), new_convbwd_op.getResult(2));
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, new_convbwd_op.getResults());
    }
    // update func result type
    module::updateModuleTypes();
    return success();
  }
};

class SwapDimMerge : public OpRewriterPatternEx<tpu::SwapDimInnerOp> {
public:
  SwapDimMerge(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::SwapDimInnerOp>(context, "SwapDimMerge",
                                                 benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::SwapDimInnerOp op,
                                    PatternRewriter &rewriter) const override {
    if (!(module::isMARS3() || module::isSGTPUV8())) {
      return failure();
    }

    auto nextOp = dyn_cast<tpu::SwapDimInnerOp>(*op.getOutput().user_begin());
    if (!nextOp || !nextOp->hasOneUse()) {
      return failure();
    }

    auto offset_lv0 = *module::getI64Array(op.getOffset());
    auto offset_lv1 = *module::getI64Array(nextOp.getOffset());
    if (offset_lv0.size() != offset_lv1.size()) {
      return failure();
    }

    int offset_nonzero_lv0 = 0;
    int num_nonzero_lv0 = 0;
    int index_lv0 = 0;
    for (int i = 0; i < offset_lv0.size(); ++i) {
      if (offset_lv0[i] > 0)
        num_nonzero_lv0 += 1;
    }
    if (num_nonzero_lv0 > 1)
      return failure();

    int offset_nonzero_lv1 = 0;
    int num_nonzero_lv1 = 0;
    int index_lv1 = 0;
    for (int i = 0; i < offset_lv1.size(); ++i) {
      if (offset_lv1[i] > 0)
        num_nonzero_lv1 += 1;
    }
    if (num_nonzero_lv1 > 1)
      return failure();

    for (int i = 0; i < offset_lv0.size(); ++i) {
      if (offset_lv0[i] > 0) {
        offset_nonzero_lv0 = offset_lv0[i];
        index_lv0 = i;
        break;
      }
    }
    for (int i = 0; i < offset_lv1.size(); ++i) {
      if (offset_lv1[i] > 0) {
        offset_nonzero_lv1 = offset_lv1[i];
        index_lv1 = i;
        break;
      }
    }
    if (std::abs(index_lv1 - index_lv0) != 1)
      return failure();

    auto shape_0 = module::getShape(op.getInput());
    auto shape_1 = module::getShape(nextOp.getInput());
    if (shape_0.size() != shape_1.size())
      return failure();
    if (shape_0[index_lv0] != shape_0[index_lv1])
      return failure();
    if (shape_1[index_lv0] != shape_1[index_lv1])
      return failure();

    std::vector<int64_t> new_offset(offset_lv0.size(), 0);
    new_offset[index_lv0] = offset_nonzero_lv0;
    new_offset[index_lv1] = offset_nonzero_lv1;

    std::vector<mlir::Value> operands(op->getOperands().begin(),
                                      op->getOperands().end());

    auto newSwapDimOp = rewriter.create<tpu::SwapDimInnerOp>(
        nextOp->getLoc(), nextOp.getOutput().getType(), operands,
        op->getAttrs());

    newSwapDimOp.setOffsetAttr(rewriter.getI64ArrayAttr(new_offset));
    rewriter.replaceOp(nextOp, newSwapDimOp.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

// reshape -> cast -> layernorm -> cast ==> cast -> layernorm -> cast -> reshape
class MoveReshapeInSubGraphPattern
    : public OpRewriterPatternEx<tpu::ReshapeOp> {
public:
  MoveReshapeInSubGraphPattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::ReshapeOp>(
            context, "MoveReshapeInSubGraphPattern", benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::ReshapeOp reshapeOp,
                                    PatternRewriter &rewriter) const override {
    if (!reshapeOp->hasOneUse()) {
      return failure();
    }

    auto output = reshapeOp.getOutput();
    if (!output) {
      return failure();
    }

    auto userIt = output.user_begin();
    if (userIt == output.user_end()) {
      return failure();
    }

    Operation *castOp = *userIt;
    if (!isa<tpu::CastOp>(castOp)) {
      return failure();
    }

    auto castOpInst = dyn_cast<tpu::CastOp>(castOp);
    if (!castOpInst || !castOpInst->hasOneUse()) {
      return failure();
    }

    auto nextUserIt = castOpInst.getOutput().user_begin();
    if (nextUserIt == castOpInst.getOutput().user_end()) {
      return failure();
    }

    Operation *nextOp = *nextUserIt;
    if (!nextOp) {
      return failure();
    }

    auto ori_loc = reshapeOp.getLoc();
    auto ishape = module::getShape(reshapeOp.getInput());
    if (isa<tpu::LayerNormOp>(nextOp)) {
      auto layerNormOpInst = dyn_cast<tpu::LayerNormOp>(nextOp);
      if (!layerNormOpInst || !layerNormOpInst->hasOneUse()) {
        return failure();
      }

      auto afterLayerNormUserIt = layerNormOpInst.getOutput().user_begin();
      if (afterLayerNormUserIt == layerNormOpInst.getOutput().user_end()) {
        return failure();
      }
      Operation *afterLayerNormOp = *afterLayerNormUserIt;
      auto layerNorm_out = layerNormOpInst.getResult();
      auto axis = layerNormOpInst.getAxis();
      ReshapeResult CanReshapeDown_Param =
          canReshapeSinkAfter(layerNormOpInst, reshapeOp);
      if (CanReshapeDown_Param.CanReshapeDown) {
        axis = CanReshapeDown_Param.out_axis;
        layerNormOpInst->setAttr("axis", rewriter.getSI32IntegerAttr(axis));
        auto gamma_weight_value = layerNormOpInst->getOperand(1);
        auto beta_weight_value = layerNormOpInst->getOperand(2);
        auto w_shape = module::getShape(gamma_weight_value);
        std::vector<int64_t> w_shape_vector(w_shape.begin(), w_shape.end());
        std::vector<int64_t> i_shape_vector(ishape.begin(), ishape.end());
        if (w_shape_vector.size() != i_shape_vector.size()) {
          int diff_len = i_shape_vector.size() - w_shape_vector.size();
          if (diff_len > 0) {
            w_shape_vector.insert(w_shape_vector.begin(), diff_len, 1);
          } else {
            w_shape_vector.erase(w_shape_vector.begin(),
                                 w_shape_vector.begin() - diff_len);
          }
        }
        module::setShape(gamma_weight_value, w_shape_vector);
        module::setShape(beta_weight_value, w_shape_vector);
        layerNormOpInst->setOperand(1, gamma_weight_value);
        layerNormOpInst->setOperand(2, beta_weight_value);
      }
      if (afterLayerNormOp && isa<tpu::CastOp>(afterLayerNormOp)) {
        auto castOpInst_2 = dyn_cast<tpu::CastOp>(afterLayerNormOp);
        if (CanReshapeDown_Param.CanReshapeDown) {
          reshapeOp.replaceAllUsesWith(reshapeOp.getInput());
          auto next_out = castOpInst_2.getResult();
          ori_loc = castOpInst_2.getLoc();
          module::setLocSuffix(castOpInst_2, "reshape_down");
          auto castOpInst_out = castOpInst.getResult();
          module::setShape(castOpInst_out, ishape);
          module::setShape(layerNorm_out, ishape);
          module::setShape(next_out, ishape);
          rewriter.setInsertionPointAfterValue(next_out);
          auto out_shape = module::getShape(reshapeOp.getOutput()).vec();
          auto reshape_type = module::getTypeLike(next_out, out_shape);
          auto shapeAttr = reshapeOp.getShape();
          auto new_reshape_op = rewriter.create<tpu::ReshapeOp>(
              ori_loc, reshape_type, ValueRange{next_out},
              rewriter.getNamedAttr("shape", shapeAttr));
          rewriter.replaceAllUsesExcept(next_out, new_reshape_op.getOutput(),
                                        new_reshape_op);
          rewriter.eraseOp(reshapeOp);
          return success();
        } else {
          return failure();
        }
      } else {
        if (CanReshapeDown_Param.CanReshapeDown) {
          reshapeOp.replaceAllUsesWith(reshapeOp.getInput());
          rewriter.setInsertionPointAfter(layerNormOpInst);
          ori_loc = layerNormOpInst.getLoc();
          module::setLocSuffix(layerNormOpInst, "reshape_down");
          auto newReshapeOp = rewriter.create<tpu::ReshapeOp>(
              ori_loc, layerNormOpInst.getResult().getType(),
              layerNormOpInst.getResult());
          module::setShape(layerNorm_out, ishape);
          auto castOpInst_out = castOpInst.getResult();
          module::setShape(castOpInst_out, ishape);
          rewriter.replaceAllUsesExcept(layerNorm_out, newReshapeOp.getOutput(),
                                        newReshapeOp);
          rewriter.eraseOp(reshapeOp);
          return success();
        } else {
          return failure();
        }
      }
    }
    return failure();
  }

private:
  struct ReshapeResult {
    bool CanReshapeDown;
    int out_axis;
  };

  ReshapeResult canReshapeSinkAfter(tpu::LayerNormOp layerNormOp,
                                    tpu::ReshapeOp reshapeOp) const {
    ReshapeResult result = {false, -1};
    auto axis = layerNormOp.getAxis();
    auto inputShape = module::getShape(reshapeOp.getInput());
    auto outputShape = module::getShape(reshapeOp.getOutput());
    std::vector<int> unchangedDims;
    size_t inputIndex = 0, outputIndex = 0;
    int64_t inputProduct = 1, outputProduct = 1;
    bool inputContinue = true, outputContinue = true;
    while (inputIndex < inputShape.size() && outputIndex < outputShape.size()) {
      if (inputContinue) {
        inputProduct *= inputShape[inputIndex];
      }
      if (outputContinue) {
        outputProduct *= outputShape[outputIndex];
      }

      if (inputProduct == outputProduct) {
        if (inputShape[inputIndex] == outputShape[outputIndex]) {
          if (inputIndex != 0 && outputIndex != 0 &&
              inputShape[inputIndex] != 1 && outputIndex == axis) {
            result.CanReshapeDown = true;
            result.out_axis = inputIndex;
            break;
          }
        }
        inputProduct = 1;
        outputProduct = 1;
        inputIndex++;
        outputIndex++;
        inputContinue = true;
        outputContinue = true;
      } else if (inputProduct < outputProduct) {
        inputIndex++;
        inputContinue = true;
        outputContinue = false;
      } else {
        outputIndex++;
        inputContinue = false;
        outputContinue = true;
      }
    }

    return result;
  }
};
struct WhereBnbwdFusePattern : public OpRewriterPatternEx<tpu::BatchNormBwdOp> {
  // using OpRewriterPatternEx::OpRewriterPatternEx;

  WhereBnbwdFusePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::BatchNormBwdOp>(
            context, "WhereBnbwdFusePattern", benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::BatchNormBwdOp op,
                                    PatternRewriter &rewriter) const override {
    if (!module::isBM1690Family())
      return failure();
    auto where_op =
        dyn_cast_or_null<tpu::WhereOp>(op.getOperand(0).getDefiningOp());
    if (!where_op)
      return failure();
    auto batchnormfwd_op = dyn_cast_or_null<tpu::BatchNormTrainOp>(
        where_op.getOperand(0).getDefiningOp());
    // auto input_shape = module::getShape(where_op.getOperand(0));
    // if(input_shape.size() != 4 || input_shape[3] < 56)
    //   return failure();
    std::vector<Value> operands;
    if (!batchnormfwd_op) {
      return failure();
      operands.push_back(where_op.getOperand(0));
    } else {
      operands.push_back(module::getNoneOp(op));
    }
    operands.push_back(where_op.getOperand(1));
    operands.push_back(op.getOperand(1));
    operands.push_back(op.getOperand(2));
    if (!batchnormfwd_op) {
      operands.push_back(module::getNoneOp(op));
    } else {
      operands.push_back(batchnormfwd_op.getOperand(4));
    }
    operands.push_back(op.getOperand(3));
    operands.push_back(op.getOperand(4));
    operands.push_back(module::getNoneOp(op));
    std::vector<Type> new_types;
    new_types.reserve(3);
    for (int i = 0; i < 3; i++) {
      new_types.push_back(op.getResult(i).getType());
    }
    auto whereBnbwdOp = rewriter.create<tpu::WhereBnbwdOp>(
        op->getLoc(), new_types, operands, op->getAttrs());
    whereBnbwdOp->setAttr("do_recompute",
                          rewriter.getBoolAttr(batchnormfwd_op != NULL));
    rewriter.replaceAllUsesWith(op->getResult(0), whereBnbwdOp.getResult(0));
    rewriter.replaceAllUsesWith(op->getResult(1), whereBnbwdOp.getResult(1));
    rewriter.replaceAllUsesWith(op->getResult(2), whereBnbwdOp.getResult(2));
    rewriter.eraseOp(op);
    rewriter.eraseOp(where_op);
    return success();
  }
};

// conv => reshape(in0)+reshape(in1)+matmul(right_transpose=true)+reshape(out)
// matmul extend do_relu from conv
// condition: kh, kw = ih, iw and pad_shape is [0,0,0,0]
struct ConvToMatMulPattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
  ConvToMatMulPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "ConvToMatMulPattern",
                                           benifit) {}

  LogicalResult matchAndRewriteImpl(tpu::Conv2DOp convOp,
                                    PatternRewriter &rewriter) const override {
    // get conv input
    auto input = convOp.getInput();
    auto filter = convOp.getFilter();
    auto bias = convOp.getBias();
    // get pad shape
    auto pad_top =
        convOp.getPadsAttr().getValue()[0].cast<mlir::IntegerAttr>().getInt();
    auto pad_left =
        convOp.getPadsAttr().getValue()[1].cast<mlir::IntegerAttr>().getInt();
    auto pad_bottom =
        convOp.getPadsAttr().getValue()[2].cast<mlir::IntegerAttr>().getInt();
    auto pad_right =
        convOp.getPadsAttr().getValue()[3].cast<mlir::IntegerAttr>().getInt();
    // pad must be 0
    if (pad_top != 0 || pad_left != 0 || pad_bottom != 0 || pad_right != 0) {
      return failure();
    }

    // group must be 1 and weight is coeff and not merged and not
    // use_3icOptimize
    if (convOp.getGroup() != 1 || convOp.getWeightIsCoeff() != true ||
        convOp.getCoeffMerged() || convOp.getUse_3icOptimize()) {
      return failure();
    }
    // dialation must be 1
    if (auto dilations = convOp.getDilations()) {
      auto values = dilations.value().getValue();
      if (values[0].cast<mlir::IntegerAttr>().getInt() != 1 ||
          values[1].cast<mlir::IntegerAttr>().getInt() != 1) {
        return failure();
      }
    }

    // use_winograd must be 0
    if (auto use_winograd = convOp.getUseWinograd()) {
      if (use_winograd.value()) {
        return failure();
      }
    }

    // multiplier must be 1
    if (auto multiplier = convOp.getMultiplier()) {
      if (multiplier.value().getValue()[0].cast<mlir::IntegerAttr>().getInt() !=
          1) {
        return failure();
      }
    }

    // rshift must be 0
    if (auto rshift = convOp.getRshift()) {
      if (rshift.value().getValue()[0].cast<mlir::IntegerAttr>().getInt() !=
          0) {
        return failure();
      }
    }

    // do_leaky_relu must be 0
    if (auto do_leaky_relu = convOp.getDoLeakyRelu()) {
      if (do_leaky_relu.value()) {
        return failure();
      }
    }

    // get input shape
    auto inputShape = module::getShape(input);
    auto filterShape = module::getShape(filter);
    // check shape.size()
    if (inputShape.size() < 4 || filterShape.size() < 4) {
      return failure();
    }
    // get nchw
    auto batchSize = inputShape[0];
    int64_t inChannel = inputShape[1];
    int64_t inHeight = inputShape[2];
    int64_t inWidth = inputShape[3];
    int64_t outChannel = filterShape[0];
    int64_t kernelHeight = filterShape[2];
    int64_t kernelWidth = filterShape[3];
    // condition: kh, kw = ih, iw
    if (kernelHeight != inHeight || kernelWidth != inWidth) {
      return failure();
    }
    if (!module::isBM1688()) {
      return failure();
    }
    auto input_type = module::getStorageType(input);
    // if (!input_type.isInteger(8)) {
    if (input_type.isF16()) {
      return failure();
    }
    // reshape input and filter
    auto inputName = module::getName(input);
    auto filterName = module::getName(filter);
    auto outputName = module::getName(convOp.getOutput());
    auto reshapeL_loc = NameLoc::get(rewriter.getStringAttr(
        inputName.str() + outputName.str() + "_matmalL_reshape"));
    auto reshapeR_loc = NameLoc::get(rewriter.getStringAttr(
        filterName.str() + outputName.str() + "_matmalR_reshape"));
    std::vector<int64_t> reshapeL_shape = {1, batchSize,
                                           inChannel * inHeight * inWidth};
    std::vector<int64_t> reshapeR_shape = {
        1, outChannel, inChannel * kernelHeight * kernelWidth};
    // std::vector<int64_t> reshapeL_shape = {1, batchSize, -1};
    // std::vector<int64_t> reshapeR_shape = {1, outChannel, -1};
    auto reshapeL_type = module::getTypeLike(input, reshapeL_shape);
    auto reshapeR_type = module::getTypeLike(filter, reshapeR_shape);
    rewriter.setInsertionPointAfter(convOp);
    auto reshapeL_op = rewriter.create<tpu::ReshapeOp>(
        reshapeL_loc, reshapeL_type, ValueRange{input});
    rewriter.setInsertionPointAfter(reshapeL_op);
    auto reshapeR_op = rewriter.create<tpu::ReshapeOp>(
        reshapeR_loc, reshapeR_type, ValueRange{filter});
    // matmul
    auto none = module::getNoneOp(convOp);
    auto matmul_loc =
        module::getLocLike(reshapeR_op.getOutput(), "conv2matmul");
    std::vector<int64_t> matmul_shape = {1, batchSize, outChannel};
    auto matmul_type = RankedTensorType::get(
        matmul_shape, module::getElementType(convOp.getOutput()));
    rewriter.setInsertionPointAfter(reshapeR_op);
    std::vector<NamedAttribute> attrs;
    // right_transpose
    attrs.push_back(
        rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(true)));
    // do_relu
    attrs.push_back(rewriter.getNamedAttr("do_relu", convOp.getDoReluAttr()));
    // relu_limit
    attrs.push_back(
        rewriter.getNamedAttr("relu_limit", convOp.getReluLimitAttr()));
    // kernel_zp
    attrs.push_back(
        rewriter.getNamedAttr("right_zp", convOp.getKernelZpAttr()));
    // quant mode
    attrs.push_back(
        rewriter.getNamedAttr("quant_mode", convOp.getQuantModeAttr()));
    // round_mode
    attrs.push_back(
        rewriter.getNamedAttr("round_mode", convOp.getRoundModeAttr()));

    auto matmul_op = rewriter.create<tpu::MatMulOp>(
        matmul_loc, matmul_type,
        ValueRange{reshapeL_op.getOutput(), reshapeR_op.getOutput(), bias, none,
                   none},
        attrs);
    // reshape output
    std::vector<int64_t> reshape_matmul_shape = {batchSize, outChannel, 1, 1};
    auto reshape_matmul_type = RankedTensorType::get(
        reshape_matmul_shape, module::getElementType(matmul_op.getOutput()));
    rewriter.setInsertionPointAfter(matmul_op);
    auto reshape_matmul_op =
        rewriter.create<tpu::ReshapeOp>(convOp.getLoc(), reshape_matmul_type,
                                        ValueRange{matmul_op.getOutput()});
    // rewrite
    rewriter.replaceAllUsesWith(convOp.getOutput(),
                                reshape_matmul_op.getOutput());
    // rewriter.replaceAllUsesExcept(convOp.getOutput(),
    // reshape_matmul_op.getOutput(), reshape_matmul_op);
    rewriter.eraseOp(convOp);
    return success();
  }
};

class DeconvPadPattern : public OpRewriterPatternEx<tpu::DeconvOp> {
public:
  // using OpRewriterPatternEx::OpRewriterPatternEx;
  DeconvPadPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::DeconvOp>(context, "DeconvPadPattern",
                                           benifit) {}
  LogicalResult matchAndRewriteImpl(tpu::DeconvOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isMARS3())
      return failure();
    auto in_shape = module::getShape(op.getInput());
    int dims = in_shape.size() - 2;
    int conv_padding_h_top = 0, conv_padding_w_left = 0;
    int conv_padding_h_bottom = 0, conv_padding_w_right = 0;
    int conv_insert_zero_x = 0, conv_insert_zero_y = 0;
    deconv_attr_t attrs = op.parseParam();
    conv_padding_h_top = attrs.kh - 1 - attrs.pad_h;
    conv_padding_w_left = attrs.kw - 1 - attrs.pad_w;
    conv_padding_h_bottom = attrs.kh - 1 - attrs.pad_h_after;
    conv_padding_w_right = attrs.kw - 1 - attrs.pad_w_after;
    auto output_shape_pad = llvm::SmallVector<int64_t>(in_shape);
    llvm::SmallVector<int64_t> pad_paddings(in_shape.size() * 2, 0);
    std::vector<int64_t> insert_zeros;
    if (dims == 3) {
      // to do convtranspose3d
      return failure();
    } else if (dims == 2) { // for convtranspose2d
      if (conv_padding_h_top > 15 || conv_padding_h_bottom > 15 ||
          conv_padding_w_left > 15 || conv_padding_w_right > 15) {
        attrs.oh = attrs.ih + conv_padding_h_top + conv_padding_h_bottom;
        attrs.ow = attrs.iw + conv_padding_w_left + conv_padding_w_right;
        if (attrs.sh > 1) {
          conv_insert_zero_y = attrs.sh - 1;
          attrs.oh = attrs.oh + conv_insert_zero_y * (attrs.ih - 1);
        }
        if (attrs.sw > 1) {
          conv_insert_zero_x = attrs.sw - 1;
          attrs.ow = attrs.ow + conv_insert_zero_x * (attrs.iw - 1);
        }
        output_shape_pad[2] = attrs.oh;
        output_shape_pad[3] = attrs.ow;
        pad_paddings[2] = conv_padding_h_top;
        pad_paddings[3] = conv_padding_w_left;
        pad_paddings[6] = conv_padding_h_bottom;
        pad_paddings[7] = conv_padding_w_right;
        insert_zeros.emplace_back(conv_insert_zero_y);
        insert_zeros.emplace_back(conv_insert_zero_x);
      } else {
        return failure();
      }
    } else if (dims == 1) { // convtranspose1d
      if (conv_padding_h_top > 15 || conv_padding_h_bottom > 15) {
        attrs.oh = attrs.ih + conv_padding_h_top + conv_padding_h_bottom;
        if (attrs.sh > 1) {
          conv_insert_zero_y = attrs.sh - 1;
          attrs.oh = attrs.oh + conv_insert_zero_y * (attrs.ih - 1);
        }
        output_shape_pad[2] = attrs.oh;
        pad_paddings[2] = conv_padding_h_top;
        pad_paddings[5] = conv_padding_h_bottom;
        insert_zeros.emplace_back(conv_insert_zero_y);
      } else {
        return failure();
      }
    } else {
      return failure();
    }
    auto output_name = module::getName(op.getInput());
    auto input_ele_type = module::getElementType(op.getInput());
    std::string name_pad = output_name.str() + "_pad";
    auto loc_pad = NameLoc::get(rewriter.getStringAttr(name_pad));
    std::vector<Value> operands_pad;
    operands_pad.push_back(op.getInput());
    operands_pad.push_back(module::getNoneOp(op));
    operands_pad.push_back(module::getNoneOp(op));
    operands_pad.push_back(module::getNoneOp(op));
    operands_pad.push_back(module::getNoneOp(op));
    std::vector<NamedAttribute> attrs_pad;
    attrs_pad.push_back(rewriter.getNamedAttr(
        "paddings", rewriter.getI64ArrayAttr(pad_paddings)));
    attrs_pad.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::PaddingModeAttr::get(getContext(), tpu::PaddingMode::constant)));
    attrs_pad.push_back(
        rewriter.getNamedAttr("with_insert_zero", rewriter.getBoolAttr(true)));
    attrs_pad.push_back(rewriter.getNamedAttr(
        "insert_zeros", rewriter.getI64ArrayAttr(insert_zeros)));
    auto op_pad = rewriter.create<tpu::PadOp>(
        loc_pad, RankedTensorType::get(output_shape_pad, input_ele_type),
        operands_pad, attrs_pad);
    std::vector<NamedAttribute> conv_attrs;
    int size = op->getAttrs().size();
    std::cout << size;
    conv_attrs.push_back(
        rewriter.getNamedAttr("kernel_shape", op.getKernelShapeAttr()));
    conv_attrs.emplace_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
    conv_attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
    conv_attrs.push_back(rewriter.getNamedAttr("group", op.getGroupAttr()));
    conv_attrs.push_back(
        rewriter.getNamedAttr("dilations", op.getDilationsAttr()));
    conv_attrs.emplace_back(
        rewriter.getNamedAttr("inserts", rewriter.getI64ArrayAttr({0, 0})));
    conv_attrs.emplace_back(
        rewriter.getNamedAttr("do_kernel_rotate", rewriter.getBoolAttr(true)));
    // conv_attrs.emplace_back(rewriter.getNamedAttr("output_padding",rewriter.getI64ArrayAttr({0,
    // 0})));
    conv_attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
    conv_attrs.push_back(
        rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    bool with_bias = !module::isNone(op.getBias());
    conv_attrs.push_back(
        rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
    conv_attrs.push_back(
        rewriter.getNamedAttr("quant_mode", op.getQuantModeAttr()));

    auto s_op = rewriter.create<tpu::Conv2DOp>(
        op->getLoc(), op.getOutput().getType(),
        ValueRange{op_pad, op.getFilter(), op.getBias()}, conv_attrs);

    op.replaceAllUsesWith(s_op.getOutput());
    rewriter.eraseOp(op);
    return success();
  }
};

// uses Lut to calculate EXP(quant(x))
// cast -> softmax -> cast ---> Quantized softmax ->  cast
class QuantizedSoftmaxPattern : public OpRewriterPatternEx<tpu::SoftmaxOp> {
public:
  QuantizedSoftmaxPattern(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<tpu::SoftmaxOp>(context, "QuantizedSoftmaxPattern",
                                            benefit) {}

  LogicalResult matchAndRewriteImpl(tpu::SoftmaxOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getLog()) {
      return failure();
    }
    if (!op.getInput().getDefiningOp() ||
        !isa<tpu::CastOp>(op.getInput().getDefiningOp()) ||
        !module::isUniformQuantized(
            op.getInput().getDefiningOp()->getOperand(0))) {
      return failure();
    }
    if (!op.getOutput().hasOneUse() ||
        !isa<tpu::CastOp>(*op.getOutput().user_begin())) {
      return failure();
    }
    auto input_shape = module::getShape(op.getInput());
    if (input_shape.size() != 4 || op.getAxis() != 2 && op.getAxis() != 3) {
      return failure();
    }
    // NOTE: here is a TEMPORARY solution, because
    // hdim-is-batch-pattern must be applied before this pattern,
    if (input_shape[1] < input_shape[2]) {
      return failure();
    }
    if (input_shape[op.getAxis()] > 10000) {
      return failure(); // avoid FP16 overflow
    }
    // NOTE: above is the TEMPORARY solution.
    if (!module::isAsymmetric()) {
      return failure();
    }
    // Rewrite:
    auto pre_cast_op = dyn_cast<tpu::CastOp>(op.getInput().getDefiningOp());
    auto beta_v = op.getBeta().convertToDouble();
    rewriter.setInsertionPointAfter(pre_cast_op);

    // ReduceMax:
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "axes", rewriter.getI64ArrayAttr({op.getAxis()})));
    attrs.push_back(
        rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceMax")));
    std::vector<int64_t> reduced_shape = input_shape;
    reduced_shape[op.getAxis()] = 1;
    auto noneOp = module::getNoneOp(op);
    auto reduceMaxOp = rewriter.create<tpu::ReduceOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(pre_cast_op.getInput()).str() + "_reduceMax")),
        module::getTypeLike(pre_cast_op.getInput(), reduced_shape),
        ValueRange{pre_cast_op.getInput(), noneOp, noneOp}, attrs);

    // BinaryShiftOp:
    attrs.clear();
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("Sub")));
    attrs.push_back(
        rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(0)));
    auto binary_shiftOp = rewriter.create<tpu::BinaryShiftOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(reduceMaxOp, 0).str() + "_binaryShift")),
        module::getTypeLike(pre_cast_op.getInput(), input_shape),
        ValueRange{pre_cast_op.getInput(), reduceMaxOp.getOutput()}, attrs);

    // LutOp:
    auto table = create_lookup_table_fp16(
        binary_shiftOp.getOutput(),
        [&beta_v](double x) { return std::exp(x * beta_v); });
    auto lutOp = rewriter.create<tpu::LutOp>(
        NameLoc::get(rewriter.getStringAttr(
            module::getName(binary_shiftOp, 0).str() + "_lut")),
        RankedTensorType::get(input_shape, rewriter.getF16Type()),
        ValueRange{binary_shiftOp.getOutput(), table});

    // ReduceSum:
    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "axes", rewriter.getI64ArrayAttr({op.getAxis()})));
    attrs.push_back(
        rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceSum")));
    auto reduceSumOp = rewriter.create<tpu::ReduceOp>(
        NameLoc::get(rewriter.getStringAttr(module::getName(lutOp, 0).str() +
                                            "_reduceSum")),
        module::getTypeLike(lutOp.getOutput(), reduced_shape),
        ValueRange{lutOp.getOutput(), noneOp, noneOp}, attrs);

    // ReciprocalOp:
    mlir::Value reciprocal_expsum;
    if (module::getChip() == module::Chip::BM1684X) {
      // Cast to F32 before reciprocal
      auto castToF32Op = rewriter.create<tpu::CastOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(reduceSumOp, 0).str() + "_toF32")),
          RankedTensorType::get(reduced_shape, rewriter.getF32Type()),
          ValueRange{reduceSumOp.getOutput()});

      // Reciprocal in F32
      auto reciprocalOp = rewriter.create<tpu::ReciprocalOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(castToF32Op, 0).str() + "_reciprocal")),
          RankedTensorType::get(reduced_shape, rewriter.getF32Type()),
          ValueRange{castToF32Op.getOutput()});
      reciprocalOp.setConstVal(APFloat(1.0));

      // Cast back to original type
      auto castBackOp = rewriter.create<tpu::CastOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(reciprocalOp, 0).str() + "_toF16")),
          module::getTypeLike(lutOp.getOutput(), reduced_shape),
          ValueRange{reciprocalOp.getOutput()});
      reciprocal_expsum = castBackOp.getOutput();
    } else {
      auto reciprocalOp = rewriter.create<tpu::ReciprocalOp>(
          NameLoc::get(rewriter.getStringAttr(
              module::getName(reduceSumOp, 0).str() + "_reciprocal")),
          module::getTypeLike(lutOp.getOutput(), reduced_shape),
          ValueRange{reduceSumOp.getOutput()});
      reciprocalOp.setConstVal(APFloat(1.0));
      reciprocal_expsum = reciprocalOp.getOutput();
    }

    // MulOp:
    rewriter.replaceOpWithNewOp<tpu::MulOp>(
        op, module::getTypeLike(lutOp.getOutput(), input_shape),
        ValueRange{lutOp.getOutput(), reciprocal_expsum});
    return success();
  }
};

namespace tpu {
using namespace bm1684x;
void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns) {
  auto ctx = patterns->getContext();
  patterns->add<MatMulRequantIntFusion>(ctx, 10);
  patterns->add<LargePadConvPattern, ConvToMatMulPattern>(ctx, 9);
  // clang-format off
  patterns->add<MatMulHdimBatchPattern,
                MatMulRemoveReshapePattern,
                MatMulLeftReusePattern,
                GroupConv2NormalConv,
                MovePermuteAfterAdd,
                TpuReshapeReorderPattern,
                PermuteAddWeightReorderPattern,
                PermuteRopeWeightReorderPattern,
                MaskedFillPermuteMove,
                PermuteFuse,
                PermuteFuse2,
                PermuteFuseAddSoftmax,
                PermuteFuseAddSoftmaxSlice,
                patterns::FuseRepeatPattern<tpu::ReshapeOp>,
                PermuteReshapeFuse,
                ReshapePermuteFuse,
                PermuteReshapeFuse2,
                GatherElementsPattern,
                ScatterElementsPattern,
                PermuteReorderPattern,
                PermutePadSwap,
                FitPermute2Hdim,
                ErasePermuteAroundAdd,
                PermuteMulconstSwap,
                MatMulActiveMatMulPattern,
                RotaryPosEmbPattern,
                ReshapeSliceSqueezePattern,
                NoneZeroFixRowMajor,
                SplitReduceL2Pattern,
                Reduce2AxesPattern,
                SplitMatmulPattern,
                GridSamplerFusePattern,
                CanCutGridSamplerFusePattern,
                TryInsertTileBinaryPattern<tpu::AddOp>,
                TryInsertTileBinaryPattern<tpu::MulOp>,
                Concat5dto4d,
                EliminateCastBeforeGatherElements,
                ConvMergeRequant,
                // CastGradWeight,
                RemoveReshape,
                MoveReshapeInSubGraphPattern,
                SwapDimMerge,
                MatMulRequantIntFusion,
                RemoveReshape,
                WhereBnbwdFusePattern,
                // ConvMergePattern
                DeconvPadPattern
                >(ctx, 8);
  // clang-format on
  patterns->add<TileMatMulHdimBatchPattern>(ctx, 7);
  patterns->add<SplitQuantizedMLP2Pattern>(ctx, 3);
  patterns->add<SplitMixedQuantizedMLPPattern>(ctx, 4);
  // patterns->add<MatmulUsePermutePattern>(ctx, 4);
  patterns->add<MultipleSameActivationMatmulMergePattern>(ctx, 3);
  patterns->add<QuantizedSoftmaxPattern>(ctx, 5);
}
} // namespace tpu

} // namespace tpu_mlir
