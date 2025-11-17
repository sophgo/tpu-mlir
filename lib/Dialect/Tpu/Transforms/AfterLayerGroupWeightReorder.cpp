//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace llvm;
using namespace tpu_mlir::backend;
namespace tpu_mlir {
namespace tpu {

static std::pair<Value, tpu::LoadOp> pickReorderWeight(Operation *op) {
  if (!op->hasAttr("ginfo")) {
    return {};
  }
  for (auto in : op->getOperands()) {
    if (auto load = in.getDefiningOp<tpu::LoadOp>()) {
      if (load.getIsIdxWeight()) {
        auto idx = load->getOperand(0);
        auto weight_op = idx.getDefiningOp<top::WeightOp>();
        if (weight_op && weight_op->hasAttr("allow_split")) {
          return {idx, load};
        }
      }
    }
  }
  return {};
}

struct UpsampleWeightReorderPattern : public OpRewritePattern<tpu::UpsampleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::UpsampleOp op,
                                PatternRewriter &rewriter) const override {

    auto picked = pickReorderWeight(op);
    if (!picked.first) {
      op->setOperand(1, module::getNoneOp(op));
      return failure();
    }

    auto input = op.getOperand(0).getDefiningOp();
    if (!input->hasAttr("ginfo"))
      return failure();
    auto in_gi = input->getAttr(LocalGenInterface::kLayerGroupAttrName)
                     .cast<tpu::LayerGroupAttr>();
    auto in_h_idx = in_gi.getHIdx().asArrayRef();
    auto in_w_idx = in_gi.getWIdx().asArrayRef();
    auto in_w_slice = in_gi.getWSlice().asArrayRef();

    auto load_op = picked.second;
    auto out_gi = load_op->getAttr(LocalGenInterface::kLayerGroupAttrName)
                      .cast<tpu::LayerGroupAttr>();
    auto h_idx = out_gi.getHIdx().asArrayRef();
    auto w_idx = out_gi.getWIdx().asArrayRef();
    auto h_slice = out_gi.getHSlice().asArrayRef();
    auto w_slice = out_gi.getWSlice().asArrayRef();

    auto v = picked.first.getDefiningOp();
    auto idx_op = dyn_cast<top::WeightOp>(v);
    auto idx_u16 = idx_op.read<uint16_t>();
    auto count = idx_u16->size();
    assert(count % 2 == 0);
    const size_t pos_num = static_cast<size_t>(count / 2);
    auto reorder = std::make_shared<std::vector<uint16_t>>();
    reorder->reserve(pos_num);
    auto ensure_size = [&](size_t need) {
      if (reorder->size() < need) {
        reorder->resize(need);
      }
    };
    SmallVector<int64_t> new_w_idx_vec;
    SmallVector<int64_t> new_w_slice_vec;
    size_t write_pos = 0;
    const int64_t out_w = w_idx.back() + w_slice.back();
    const size_t row_stride = static_cast<size_t>(out_w);
    for (size_t hi = 0; hi < h_idx.size(); ++hi) {
      size_t h_base = write_pos;
      const int64_t h0 = h_idx[hi];
      const int64_t hlen = h_slice[hi];
      for (size_t wi = 0; wi < w_idx.size(); ++wi) {
        const int64_t w0 = w_idx[wi];
        const int64_t wlen = w_slice[wi];
        const int64_t param_w_slice = in_w_slice[wi];
        size_t base = write_pos;
        for (int64_t dh = 0; dh < hlen; ++dh) {
          const int64_t H = h0 + dh;
          const size_t row_base = static_cast<size_t>(H) * row_stride;
          size_t src_pair = (row_base + static_cast<size_t>(w0)) * 2;
          size_t dst_base = write_pos;
          const size_t copy_pairs = static_cast<size_t>(wlen);
          ensure_size(dst_base + copy_pairs);
          for (size_t k = 0; k < copy_pairs; ++k) {
            uint16_t h_g = (*idx_u16)[src_pair + 0];
            uint16_t w_g = (*idx_u16)[src_pair + 1];
            uint16_t h_l = h_g - static_cast<uint16_t>(in_h_idx[hi]);
            uint16_t w_l = w_g - static_cast<uint16_t>(in_w_idx[wi]);
            uint16_t idx1d = h_l * static_cast<uint16_t>(param_w_slice) + w_l;
            (*reorder)[dst_base + 0] = idx1d;
            src_pair += 2;
            dst_base += 1;
          }
          write_pos += copy_pairs;
        }
        size_t slice = write_pos - base;
        size_t padded = align_up(slice, Arch::NPU_NUM);
        size_t pad_size = padded - slice;
        if (pad_size > 0) {
          size_t pad_base = write_pos;
          ensure_size(pad_base + pad_size);
          for (size_t t = 0; t < pad_size; ++t) {
            (*reorder)[pad_base + t] = 0;
          }
          write_pos += pad_size;
        }
      }
      int64_t w_idx_block = static_cast<int64_t>(h_base / Arch::NPU_NUM);
      int64_t w_slice_block =
          static_cast<int64_t>((write_pos - h_base) / Arch::NPU_NUM);
      new_w_idx_vec.push_back(w_idx_block);
      new_w_slice_vec.push_back(w_slice_block);
    }
    int64_t total_data = static_cast<int64_t>(write_pos);
    if (reorder->size() != total_data) {
      reorder->resize(total_data);
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(idx_op);
    auto elem_type = module::getStorageType(idx_op);
    SmallVector<int64_t, 4> reorder_shape;
    auto c_per_npu = ceiling_func(total_data, Arch::NPU_NUM);
    reorder_shape = {1, c_per_npu, 1, Arch::NPU_NUM};
    auto new_type = RankedTensorType::get(reorder_shape, elem_type);
    auto new_idx = top::WeightOp::create<uint16_t>(idx_op, "reordered",
                                                   *reorder, new_type);
    auto new_idx_op = dyn_cast<top::WeightOp>(new_idx.getDefiningOp());
    auto new_idx_data = new_idx_op.read<uint16_t>();
    auto reorder_trans = std::make_shared<std::vector<uint16_t>>(total_data, 0);
    function_permute(new_idx_data->data(), reorder_trans->data(),
                     module::getShape(new_idx_op), {0, 3, 2, 1});
    SmallVector<int64_t, 4> trans_shape = {1, Arch::NPU_NUM, 1, c_per_npu};
    auto trans_type = RankedTensorType::get(trans_shape, elem_type);
    auto trans_idx = top::WeightOp::create<uint16_t>(
        idx_op, "reordered_trans", *reorder_trans, trans_type);
    auto attrs = idx_op->getAttrs();
    auto trans_idx_op = trans_idx.getDefiningOp();
    for (auto &attr : attrs) {
      auto attr_name = attr.getName().str();
      if (attr_name == "inline_bytes") {
        continue;
      }
      trans_idx_op->setAttr(attr.getName(), attr.getValue());
    }
    load_op->setOperand(0, trans_idx);
    auto out_type = trans_idx.getType().cast<RankedTensorType>();
    auto load_out = load_op.getResult();
    load_out.setType(out_type);
    op->setOperand(1, load_out);
    rewriter.eraseOp(v);
    return success();
  }
};

class AfterLayerGroupWeightReorderPass
    : public AfterLayerGroupWeightReorderBase<
          AfterLayerGroupWeightReorderPass> {
public:
  AfterLayerGroupWeightReorderPass() {}
  void runOnOperation() override {
    auto modules = module::getAllModules();
    MLIRContext &ctx = getContext();

    for (auto s : *modules) {
      for (auto func : s.getOps<FuncOp>()) {
        RewritePatternSet patterns(&ctx);
        patterns.add<UpsampleWeightReorderPattern>(&ctx);
        auto config = GreedyRewriteConfig();
        config.maxIterations = 1; // apply each pattern only once.
        applyPatternsAndFoldGreedily(func, std::move(patterns), config);
      }
    }

    module::removeUnusedOp();
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>>
createAfterLayerGroupWeightReorderPass() {
  return std::make_unique<AfterLayerGroupWeightReorderPass>();
}
} // namespace tpu
} // namespace tpu_mlir
