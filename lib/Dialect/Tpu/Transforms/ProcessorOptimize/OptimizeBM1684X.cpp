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
#include "tpu_mlir/Dialect/Tpu/Transforms/RewritePattern.inc"

using namespace llvm;
using namespace tpu_mlir::backend;
namespace tpu_mlir {

namespace bm1684x {
class MatMulHdimBatchPattern : public OpRewritePattern<tpu::MatMulOp> {
  // Case1: Permute -> MatMul <- Permute
  // Case2: Reshape -> MatMul <- Permute
  // Case3: Left    -> MatMul <- Permute
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {

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
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(out_order)));
    auto trans_op = rewriter.create<tpu::PermuteOp>(
        ori_loc, trans_type, ValueRange{mat_out, module::getNoneOp(op)}, attrs);
    rewriter.replaceAllUsesExcept(mat_out, trans_op->getResult(0), trans_op);
    return success();
  }
};

class MatMulLeftReusePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
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
class MatMulRemoveReshapePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
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

// transform group conv to normal conv, when int8/f16/bf16 &&
// input_c<=ic_parallel && isBM1684XFamily()
class GroupConv2NormalConv : public OpRewritePattern<tpu::Conv2DOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::Conv2DOp op,
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
class PermuteAddWeightReorderPattern : public OpRewritePattern<tpu::PermuteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
      add_op.getOutput().replaceAllUsesExcept(new_op.getOutput(), {new_op});
      rewriter.eraseOp(op);
      return success();
    } else if (auto mul_op = dyn_cast<tpu::MulOp>(nextOp)) {
      auto inB = mul_op.getInputs()[1];
      if (!module::isWeight(inB)) {
        return failure();
      }
      auto inB_shape = module::getShape(inB);
      if (inB_shape[1] != 1) {
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
      mul_out.replaceAllUsesExcept(new_op.getOutput(), {new_op});
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

// reorder op when transpose is before mulconst
// permute order = {0,2,3,1}
class PermuteMulconstSwap : public OpRewritePattern<tpu::PermuteOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
      mulconst_or_mulshift_out.replaceAllUsesExcept(new_op.getOutput(),
                                                    {new_op});
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
class MaskedFillPermuteMove : public OpRewritePattern<tpu::MaskedFillOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MaskedFillOp op,
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
 * permute \
 *          => Add => Add -> permute
 * permute /
 */
class MovePermuteAfterAdd : public OpRewritePattern<tpu::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto l_permute_op = op.getOperand(0).getDefiningOp<tpu::PermuteOp>();
    auto r_permute_op = op.getOperand(1).getDefiningOp<tpu::PermuteOp>();
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
    output.replaceAllUsesExcept(new_permute_op.getOutput(), new_permute_op);
    return success();
  }
};

/**
 * reshape \
 *          => Add => Add -> reshape
 * reshape /
 *
 * NOTE: may have performance problem, for example:
 *  reshape(* -> 1,64,1,1) \
 *                          => Add(1,64,1,1) => Add(1,1,1,64) -> reshape
 *  reshape(* -> 1,64,1,1) /
 *
 * Optimized pattern can not make full use of lanes.
 *
 */
class MoveReshapeAfterAdd : public OpRewritePattern<tpu::AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto l_reshape_op = op.getOperand(0).getDefiningOp<tpu::ReshapeOp>();
    auto r_reshape_op = op.getOperand(1).getDefiningOp<tpu::ReshapeOp>();
    if (!l_reshape_op || !r_reshape_op)
      return failure();
    auto l_in_shape = module::getShape(l_reshape_op.getInput()).vec();
    auto r_in_shape = module::getShape(r_reshape_op.getInput()).vec();
    if (l_in_shape != r_in_shape)
      return failure();
    auto l_out_shape = module::getShape(l_reshape_op.getOutput()).vec();
    auto r_out_shape = module::getShape(r_reshape_op.getOutput()).vec();
    if (l_out_shape != r_out_shape)
      return failure();
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
    output.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);
    return success();
  }
};

// reorder op when reshapeOp is before matmul/mulconst/cast/softmax op to
// eliminate reshapeOp
// copied from lib/Dialect/Top/Transforms/ProcessorOptimize/OptimizeBM1684X.cpp
class TpuReshapeReorderPattern : public OpRewritePattern<tpu::ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto next_op_ = *output.user_begin();

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
      next_out.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);
      rewriter.eraseOp(op);
      rewriter.eraseOp(right_op);
      return success();
    } else if (isa<tpu::MulConstOp, tpu::CastOp, tpu::SoftmaxOp>(next_op_)) {
      // check input is Reshape(n, c, h, w) --> (nxc, h, w)
      auto ishape = SmallVector<int64_t>(module::getShape(op.getInput()));
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
      next_out.replaceAllUsesExcept(new_reshape_op.getOutput(), new_reshape_op);

      if (auto next_op = dyn_cast<tpu::SoftmaxOp>(next_op_)) {
        next_op->setAttr("axis", rewriter.getSI32IntegerAttr(3));
      }
      rewriter.eraseOp(op);
      return success();
    } else if (auto next_op = dyn_cast<tpu::ReshapeOp>(next_op_)) {
      auto ishape = module::getShape(op.getInput());
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
struct PermuteFuse : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (auto rop = dyn_cast<tpu::ReshapeOp>(in.getDefiningOp())) {
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
    return success();
  }
};

// permute1 + permute2
// permute_order[0,2,1,3]!=  permute2_order[0,1,3,2]
struct PermuteFuse2 : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
    // if (in0_order == in1_order || in0_order->size() != in1_order->size()) {
    //   return failure();
    // }
    // strict restrictions
    if (false == (in1_order->size() == 4 && in1_order->at(0) == 0 &&
                  in1_order->at(1) == 1 && in1_order->at(2) == 3 &&
                  in1_order->at(3) == 2) ||
        false == (in0_order->size() == 4 && in0_order->at(0) == 0 &&
                  in0_order->at(1) == 2 && in0_order->at(2) == 1 &&
                  in0_order->at(3) == 3)) {
      return failure();
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
struct GatherElementsPattern : public OpRewritePattern<tpu::GatherElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GatherElementsOp op,
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

struct ScatterElementsPattern
    : public OpRewritePattern<tpu::ScatterElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ScatterElementsOp op,
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
struct PermuteFuseAddSoftmax : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
    auto mask_shape = module::getShape(add_op->getOperand(1));
    auto mask_name = module::getName(add_op->getOperand(1)).str();
    if (mask_shape[1] != 1) {
      return failure();
    }
    auto new_mask_type = RankedTensorType::get(
        {mask_shape[0], mask_shape[2], mask_shape[1], mask_shape[3]},
        module::getElementType(out));
    if (mask_shape[1] == 1 && mask_shape[2] == 1) {
      // nothing to do
    } else {
      auto reshape_op = rewriter.create<tpu::ReshapeOp>(
          NameLoc::get(rewriter.getStringAttr(mask_name + "_reshape")),
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

// reshape + permute -> permute
struct ReshapePermuteFuse : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
struct PermuteReshapeFuse : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
} // namespace bm1684x

//  reshape + permute + reshape + permute -> reshape + permute
//            3D(0,2,1) 6D        6D case1:(0,2,4,3,5,1)
//                                   case2:(0,2,4,1,3,5)
struct PermuteReshapeFuse2 : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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

struct FitPermute2Hdim : public OpRewritePattern<tpu::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::MatMulOp op,
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

struct ErasePermuteAroundAdd : public OpRewritePattern<tpu::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::AddOp op,
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
      out_permute_op.getOutput().replaceAllUsesExcept(op.getOutput(), op);

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
      out_permute_op.getOutput().replaceAllUsesExcept(op.getOutput(), op);

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
class TileMatMulHdimBatchPattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
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
class MarkRedundancySlicePattern : public OpRewritePattern<tpu::SliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::SliceOp op,
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
class MatMulActiveMatMulPattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
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

class RotaryPosEmbPattern : public OpRewritePattern<tpu::PermuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::PermuteOp op,
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
    unsqueeze_0_op.getOutput().replaceAllUsesExcept(
        reshape_after_unsqueeze_0_op.getOutput(), reshape_after_unsqueeze_0_op);
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
    unsqueeze_1_op.getOutput().replaceAllUsesExcept(
        reshape_after_unsqueeze_1_op.getOutput(), reshape_after_unsqueeze_1_op);
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
    concat_0_op.getOutput().replaceAllUsesExcept(
        reshape_after_concat_0_op.getOutput(), reshape_after_concat_0_op);
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
    concat_1_op.getOutput().replaceAllUsesExcept(permute_op.getOutput(),
                                                 permute_op);
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
class ReshapeSliceSqueezePattern : public OpRewritePattern<tpu::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::ReshapeOp op,
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
        if (ax > 0 && i >= ax) {
          inner_size *= in_shape[i];
        }
      }
      if (ax < 0 || steps->at(ax) != 1 || inner_size != shape[ax]) {
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

class MatMul2FAttentionPattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    std::vector<Operation *> op_need_del;
    if (!module::isBM1684X())
      return failure();
    auto out_type = module::getStorageType(op.getOutput());
    if (!out_type.isBF16() && !out_type.isF16()) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }

    // forward
    bool qm_one = false;
    tpu::ReshapeOp reshape_op;
    auto o_permute =
        dyn_cast<tpu::PermuteOp>(*(op.getOutput().getUsers().begin()));
    // (*(op.getOutput().getUsers().begin()))->dump();
    if (!o_permute) {
      reshape_op =
          dyn_cast<tpu::ReshapeOp>(*(op.getOutput().getUsers().begin()));
      if (!reshape_op) {
        return failure();
      }
      auto oshape = module::getShape(reshape_op.getOutput());
      if (oshape.size() != 3 || oshape[1] != 1) {
        return failure();
      }
      qm_one = true;
    } else {
      if (!o_permute->hasOneUse()) {
        return failure();
      }
      reshape_op =
          dyn_cast<tpu::ReshapeOp>(*(o_permute.getOutput().getUsers().begin()));
    }
    if (!reshape_op || !reshape_op->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(reshape_op);
    if (o_permute) {
      op_need_del.emplace_back(o_permute);
    }
    op_need_del.emplace_back(op);

    // backward
    tpu::SoftmaxOp softmax;
    if (auto cast_op = dyn_cast<tpu::CastOp>(op.getInput().getDefiningOp())) {
      if (!cast_op->hasOneUse()) {
        return failure();
      }
      softmax = dyn_cast<tpu::SoftmaxOp>(cast_op.getInput().getDefiningOp());
      op_need_del.emplace_back(cast_op);
    } else {
      softmax = dyn_cast<tpu::SoftmaxOp>(op.getInput().getDefiningOp());
    }
    if (!softmax || !softmax->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(softmax);
    Value mul_out;
    tpu::AddOp add;
    if (auto cast_op =
            dyn_cast<tpu::CastOp>(softmax.getInput().getDefiningOp())) {
      if (!cast_op->hasOneUse()) {
        return failure();
      }
      add = dyn_cast<tpu::AddOp>(cast_op.getInput().getDefiningOp());
      op_need_del.emplace_back(cast_op);
    } else {
      add = dyn_cast<tpu::AddOp>(softmax.getInput().getDefiningOp());
    }
    if (!add) {
      mul_out = softmax.getInput();
    } else {
      mul_out = add.getInputs()[0];
      op_need_del.emplace_back(add);
    }
    auto mul_const = dyn_cast<tpu::MulConstOp>(mul_out.getDefiningOp());
    if (!mul_const || !mul_const->hasOneUse()) {
      return failure();
    }
    op_need_del.emplace_back(mul_const);
    auto matmul0 =
        dyn_cast<tpu::MatMulOp>(mul_const.getInput().getDefiningOp());
    if (!matmul0) {
      return failure();
    }
    op_need_del.emplace_back(matmul0);
    // queries
    Value q_in;
    if (!qm_one) {
      auto q_permute =
          dyn_cast<tpu::PermuteOp>(matmul0.getInput().getDefiningOp());
      if (!q_permute || !q_permute->hasOneUse()) {
        return failure();
      }
      op_need_del.emplace_back(q_permute);
      q_in = q_permute.getInput();
    } else {
      auto q_reshape =
          dyn_cast<tpu::ReshapeOp>(matmul0.getInput().getDefiningOp());
      if (!q_reshape || !q_reshape->hasOneUse()) {
        return failure();
      }
      op_need_del.emplace_back(q_reshape);
      q_in = q_reshape.getInput();
    }

    // keys
    auto k_permute =
        dyn_cast<tpu::PermuteOp>(matmul0.getRight().getDefiningOp());
    if (!k_permute || !k_permute->hasOneUse())
      return failure();
    op_need_del.emplace_back(k_permute);

    // values
    auto v_permute = dyn_cast<tpu::PermuteOp>(op.getRight().getDefiningOp());
    if (!v_permute || !v_permute->hasOneUse())
      return failure();
    op_need_del.emplace_back(v_permute);

    rewriter.setInsertionPointAfter(reshape_op);
    auto o_shape = module::getShape(op.getOutput());
    auto sf_shape = module::getShape(softmax.getInput());
    auto none = module::getNoneOp(op);
    int64_t head;
    int64_t d;
    int64_t mq;
    int64_t mk;
    int64_t batch;

    assert(o_shape.size() == 4 && sf_shape.size() == 4);
    batch = o_shape[0];
    head = o_shape[1];
    d = o_shape[3];
    mq = sf_shape[2];
    mk = sf_shape[3];
    assert(o_shape[2] == mq && sf_shape[1] == head);

    // ppl flash attention only support d <= 256, bf16 & fp16
    if (d > 160 || mk < 4) {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("scale", mul_const.getConstValAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("head", rewriter.getI64IntegerAttr(head)));
    attrs.push_back(
        rewriter.getNamedAttr("dim", rewriter.getI64IntegerAttr(d)));
    attrs.push_back(
        rewriter.getNamedAttr("batch", rewriter.getI64IntegerAttr(batch)));
    attrs.push_back(
        rewriter.getNamedAttr("mq", rewriter.getI64IntegerAttr(mq)));
    attrs.push_back(
        rewriter.getNamedAttr("mk", rewriter.getI64IntegerAttr(mk)));
    std::vector<Value> operands;
    operands.push_back(q_in);
    operands.push_back(k_permute.getInput());
    operands.push_back(v_permute.getInput());
    operands.push_back(add ? add.getInputs()[1] : none);
    operands.push_back(none);
    auto attention = rewriter.create<tpu::FAttentionOp>(
        reshape_op.getLoc(), reshape_op.getOutput().getType(), operands, attrs);
    reshape_op.replaceAllUsesWith(attention.getOperation());
    for (auto op : op_need_del) {
      rewriter.eraseOp(op);
    }
    return success();
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

class ConvMergePattern : public OpRewritePattern<tpu::Conv2DOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::Conv2DOp op,
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

class NoneZeroFixRowMajor : public OpRewritePattern<tpu::NonZeroOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::NonZeroOp op,
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
    op.getOutput().replaceAllUsesExcept(permute_op.getOutput(), permute_op);
    return success();
  }
};

class SplitReduceL2Pattern : public OpRewritePattern<tpu::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    /* ReduceL2(1x6x8x65536)->Reshape(1x48x256x256)+ReduceL2(1x48x256x1)+ReduceL2(1x48x1x1)+Reshape(1x6x8x1)
     */
    // TODO : support quant type; consider the divisor of the reduced dim;
    auto mode = op.getMode();
    auto input = op.getInput();
    auto output = op.getOutput();
    auto input_shape = module::getShape(input);
    if (mode != "ReduceL2" || input_shape.size() != 4) {
      return failure();
    }
    auto axes = module::getI64Array(op.getAxes());
    int split_dim = input_shape[3];
    if (axes->size() != 1 || axes->at(0) != 3 || split_dim != 65536)
      return failure();

    auto name = module::getName(input);
    std::vector<Value> operands;

    std::vector<int64_t> reshape0_shape = {
        input_shape[0], input_shape[1] * input_shape[2], 256, 256};
    auto reshape0_type = module::getTypeLike(output, reshape0_shape);
    auto loc_reshape0 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape0"));
    rewriter.setInsertionPointAfterValue(input);
    auto reshape_op0 = rewriter.create<tpu::ReshapeOp>(
        loc_reshape0, reshape0_type, ValueRange{input});

    std::vector<int64_t> reducel2_0_shape = {
        input_shape[0], input_shape[1] * input_shape[2], 256, 1};
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
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceL2")));
    attrs.push_back(rewriter.getNamedAttr("axes", op.getAxes()));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reducel2_op0 = rewriter.create<tpu::ReduceOp>(
        loc_reduce0, reducel2_type0, operands, attrs);

    operands.clear();
    attrs.clear();
    std::vector<int64_t> reducel2_1_shape = {input_shape[0], input_shape[1],
                                             input_shape[2], 1};
    auto reducel2_type1 = module::getTypeLike(output, reducel2_1_shape);
    auto loc_reduce1 =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reduce1"));
    rewriter.setInsertionPointAfter(reducel2_op0);
    operands.push_back(reducel2_op0.getOutput());
    for (int i = operands.size(); i < 3; i++) {
      operands.push_back(noneOp);
    }
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceL2")));
    op.setAxesAttr(rewriter.getI64ArrayAttr({axes->at(0) - 1}));
    attrs.push_back(rewriter.getNamedAttr("axes", op.getAxes()));
    attrs.push_back(rewriter.getNamedAttr(
        "keepdims", rewriter.getBoolAttr(op.getKeepdims())));
    auto reducel2_op1 = rewriter.create<tpu::ReduceOp>(
        loc_reduce1, reducel2_type1, operands, attrs);

    std::vector<int64_t> reshape1_shape = {input_shape[0], input_shape[1],
                                           input_shape[2], 1};
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

/**
 *                 Slice
 *  \                   \
 *    MatMul   ->         MatMul => Add
 *  /                   /
 *                 Slice
 */
class SplitMatmulPattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    /*
     * MatMul(1x1x48x65536,1x1x65536x48) -> Slice =>
     * MatMul(1x1x48x16384,1x1x16384x48) => Add
     */
    // TODO : judge whether K is splited; what if other shape > 65535
    if (module::isBM1690Family()) {
      return failure();
    }
    auto left = op.getInput();
    auto right = op.getRight();
    auto left_shape = module::getShape(left);
    auto right_shape = module::getShape(right);
    auto output_shape = module::getShape(op.getOutput());
    int left_dim = left_shape.size();
    int right_dim = right_shape.size();
    auto l_trans = op.getLeftTranspose();
    auto r_trans = op.getRightTranspose();

    if (left_dim != 4 || left_shape[3] < 16384 || l_trans || !r_trans) {
      return failure();
    }
    int left_K_size = left_shape[3];
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
    std::vector<int64_t> new_left_shapes = {
        left_shape[0], left_shape[1], left_shape[2], left_shape[3] / secs};
    std::vector<int64_t> new_right_shapes = {
        right_shape[0], right_shape[1], right_shape[2], right_shape[3] / secs};
    std::vector<NamedAttribute> attrs;
    for (int i = 0; i < secs; i++) {
      left_offset[3] = left_K_size * i / secs;
      left_ends[3] = left_K_size * (i + 1) / secs;
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(3)));
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
      right_offset[3] = left_K_size * i / secs;
      right_ends[3] = left_K_size * (i + 1) / secs;
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(3)));
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
      rewriter.setInsertionPointAfter(right_op);
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

struct GridSamplerFusePattern : public OpRewritePattern<tpu::GridSamplerOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::GridSamplerOp op,
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

namespace tpu {
using namespace bm1684x;
void populateOptimizeBM1684XPatterns(RewritePatternSet *patterns) {
  auto ctx = patterns->getContext();
  if (module::getDeviceNum() < 2) {
    // DevParallel not support FAttention, need to support in future
    patterns->add<MatMul2FAttentionPattern>(ctx, 10);
  }
  patterns->add<LargePadConvPattern>(ctx, 9);
  // clang-format off
  patterns->add<MatMulHdimBatchPattern,
                MatMulRemoveReshapePattern,
                MatMulLeftReusePattern,
                GroupConv2NormalConv,
                MovePermuteAfterAdd,
                MoveReshapeAfterAdd,
                TpuReshapeReorderPattern,
                PermuteAddWeightReorderPattern,
                MaskedFillPermuteMove,
                PermuteFuse,
                PermuteFuse2,
                PermuteFuseAddSoftmax,
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
                SplitMatmulPattern,
                GridSamplerFusePattern
                // ConvMergePattern
                >(ctx, 8);
  // clang-format on
  patterns->add<TileMatMulHdimBatchPattern>(ctx, 7);
  patterns->add<SplitQuantizedMLPPattern, SplitMixedQuantizedMLPPattern>(ctx);
}
} // namespace tpu

} // namespace tpu_mlir
