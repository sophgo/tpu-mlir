// SPDX-License-Identifier: BSD-2-Clause
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include <algorithm>

namespace tpu_mlir {
namespace tpu {
namespace {

// The deploy option promises B=1, x_length=T, T<=256. A static upper
// bound alone cannot justify sharing shapes or removing a padding mask.
bool hasShape(Value v, ArrayRef<int64_t> shape) {
  auto type = v.getType().dyn_cast<RankedTensorType>();
  return type && type.getShape() == shape;
}
bool boolAttr(Operation *op, StringRef name) {
  auto value = op->getAttrOfType<BoolAttr>(name);
  return value && value.getValue();
}
bool scalar(Operation *op, StringRef name, double value) {
  auto attr = op->getAttrOfType<FloatAttr>(name);
  return attr && attr.getValueAsDouble() == value;
}
bool constant(Value v, float number) {
  auto op = v.getDefiningOp<top::WeightOp>();
  if (!op || !hasShape(v, {1}))
    return false;
  return op.read_as_float()->at(0) == number;
}
bool isValidMask(Value mask) {
  auto unsq = mask.getDefiningOp<top::UnsqueezeOp>();
  if (!unsq)
    return false;
  auto cmp = unsq->getOperand(0).getDefiningOp<top::CompareOp>();
  if (!cmp || cmp.getMode() != "Less")
    return false;
  auto cast = cmp->getOperand(0).getDefiningOp<top::CastOp>();
  auto limit = cmp->getOperand(1).getDefiningOp<top::UnsqueezeOp>();
  if (!cast || !limit || cast.getTo() != "INT32")
    return false;
  auto range = cast->getOperand(0).getDefiningOp<top::RangeOp>();
  if (!range || !constant(range->getOperand(0), 0) ||
      !constant(range->getOperand(2), 1))
    return false;
  auto end = range->getOperand(1).getDefiningOp<top::CastOp>();
  if (!end || end->getOperand(0) != limit->getOperand(0))
    return false;
  auto add = end->getOperand(0).getDefiningOp<top::AddConstOp>();
  return add && scalar(add, "const_val", 4) &&
         add->getOperand(0).getDefiningOp<top::InputOp>() &&
         module::getName(add->getOperand(0)) == "x_length";
}
bool specializeTop(func::FuncOp func) {
  Block &body = func.getBody().front();
  Value commonShape[3], maskView, mask, firstMasked;
  unsigned views = 0, products = 0, wheres = 0;
  for (auto &op : llvm::make_early_inc_range(body)) {
    if (auto view = dyn_cast<top::ReshapeOp>(op)) {
      if (view->getNumOperands() != 2)
        continue;
      auto name = module::getName(view.getOperation());
      if (!name.contains("/self_attn/Reshape"))
        continue;
      Value x = view->getOperand(0), y = view->getResult(0);
      unsigned group;
      if (hasShape(x, {1, 260, 512}) && hasShape(y, {1, 260, 4, 128}))
        group = 0;
      else if (hasShape(x, {1, 260, 4, 128}) && hasShape(y, {1, 260, 512}))
        group = 1;
      else if (hasShape(x, {1, 1, 260}) && hasShape(y, {1, 260, 1}))
        group = 2;
      else
        return false;
      ++views;
      if (group == 2) {
        if (!maskView) {
          if (!isValidMask(x))
            return false;
          mask = x;
          maskView = y;
        } else {
          if (x != mask)
            return false;
          y.replaceAllUsesWith(maskView);
          view.erase();
          continue;
        }
      }
      if (!commonShape[group])
        commonShape[group] = view->getOperand(1);
      view->setOperand(1, commonShape[group]);
    } else if (auto mul = dyn_cast<top::MulOp>(op)) {
      if (!maskView || mul->getNumOperands() != 2 ||
          mul->getOperand(1) != maskView)
        continue;
      if (boolAttr(mul, "do_relu") ||
          mul->getOperand(0).getType() != mul.getOutput().getType())
        return false;
      if (!firstMasked) {
        firstMasked = mul.getOutput();
        continue;
      }
      mul.getOutput().replaceAllUsesWith(mul->getOperand(0));
      mul.erase();
      ++products;
    } else if (auto where = dyn_cast<top::WhereOp>(op)) {
      auto cmp = where->getOperand(0).getDefiningOp<top::CompareConstOp>();
      if (!cmp || cmp.getMode() != "Equal" || !scalar(cmp, "const_val", 0))
        return false;
      auto unsq = cmp->getOperand(0).getDefiningOp<top::UnsqueezeOp>();
      if (!unsq || !mask || unsq->getOperand(0) != mask ||
          !boolAttr(where, "x_is_const") || boolAttr(where, "y_is_const") ||
          where->getOperand(2).getType() != where.getOutput().getType())
        return false;
      where.getOutput().replaceAllUsesWith(where->getOperand(2));
      where.erase();
      ++wheres;
    }
  }
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(body))) {
    if (op.use_empty() &&
        isa<top::WeightOp, top::ShapeOp, top::SliceOp, top::ConcatOp,
            top::UnsqueezeOp, top::CompareConstOp, top::ReshapeOp>(op))
      op.erase();
  }
  return views == 350 && products == 139 && wheres == 140;
}

RankedTensorType shaped(Value v, ArrayRef<int64_t> dims) {
  return RankedTensorType::get(dims, module::getElementType(v));
}
Location suffix(OpBuilder &b, Operation *op, StringRef tag) {
  return NameLoc::get(b.getStringAttr((module::getName(op) + "_" + tag).str()));
}
Value transposeHeads(OpBuilder &b, Operation *before, Value x,
                     ArrayRef<int64_t> dims, StringRef tag) {
  return b
      .create<tpu::PermuteOp>(suffix(b, before, tag), shaped(x, dims),
                              ValueRange{x, module::getNoneOp(before)},
                              ArrayRef<NamedAttribute>{b.getNamedAttr(
                                  "order", b.getI64ArrayAttr({0, 2, 1, 3}))})
      .getOutput();
}

bool headMajor(tpu::MatMulOp qk) {
  if (!qk.getOutput().hasOneUse())
    return false;
  auto soft = dyn_cast<tpu::SoftmaxOp>(*qk.getOutput().getUsers().begin());
  if (!soft || soft.getAxis() != 3 || !soft.getOutput().hasOneUse())
    return false;
  auto av = dyn_cast<tpu::MatMulOp>(*soft.getOutput().getUsers().begin());
  if (!av || av->getOperand(0) != soft.getOutput() ||
      !boolAttr(av, "hdim_is_batch") || boolAttr(av, "right_transpose") ||
      !boolAttr(qk, "right_transpose"))
    return false;
  for (auto mm : {qk, av}) {
    if (boolAttr(mm, "left_transpose") || boolAttr(mm, "output_transpose") ||
        !module::isNone(mm->getOperand(2)))
      return false;
  }
  if (!hasShape(qk->getOperand(0), {1, 260, 4, 128}) ||
      !hasShape(qk->getOperand(1), {1, 260, 4, 128}) ||
      !hasShape(av->getOperand(1), {1, 260, 4, 128}) ||
      !hasShape(qk.getOutput(), {1, 260, 4, 260}) ||
      !hasShape(av.getOutput(), {1, 260, 4, 128}))
    return false;
  OpBuilder b(qk);
  for (unsigned i = 0; i < 2; ++i)
    qk->setOperand(i, transposeHeads(b, qk, qk->getOperand(i), {1, 4, 260, 128},
                                     i ? "headk" : "headq"));
  qk->setAttr("hdim_is_batch", b.getBoolAttr(false));
  auto scores = shaped(qk.getOutput(), {1, 4, 260, 260});
  qk.getOutput().setType(scores);
  soft.getOutput().setType(scores);
  b.setInsertionPoint(av);
  av->setOperand(
      1, transposeHeads(b, av, av->getOperand(1), {1, 4, 260, 128}, "headv"));
  av->setAttr("hdim_is_batch", b.getBoolAttr(false));
  av.getOutput().setType(shaped(av.getOutput(), {1, 4, 260, 128}));
  b.setInsertionPointAfter(av);
  Value seq =
      transposeHeads(b, av, av.getOutput(), {1, 260, 4, 128}, "sequence");
  av.getOutput().replaceAllUsesExcept(seq, seq.getDefiningOp());
  return true;
}

// Split N only: every output retains the original K reduction and fused
// bias/ReLU.
Value halfWeight(top::WeightOp w, const std::vector<uint16_t> &raw,
                 unsigned part, Operation *before) {
  auto dims = module::getShape(w.getOutput()).vec();
  size_t n = dims.back(), half = n / 2;
  dims.back() = half;
  std::vector<uint16_t> result(raw.size() / 2);
  for (size_t row = 0; row < raw.size() / n; ++row)
    std::copy_n(raw.begin() + row * n + part * half, half,
                result.begin() + row * half);
  auto type = shaped(w.getOutput(), dims);
  auto value =
      top::WeightOp::create(w, part ? "split1" : "split0", result, type);
  value.getDefiningOp()->moveBefore(before);
  return value;
}
bool splitProjection(tpu::MatMulOp mm) {
  auto w = mm->getOperand(1).getDefiningOp<top::WeightOp>();
  auto b = mm->getOperand(2).getDefiningOp<top::WeightOp>();
  if (!w || !b || !w.getOutput().hasOneUse() || !b.getOutput().hasOneUse() ||
      boolAttr(mm, "right_transpose") || boolAttr(mm, "left_transpose") ||
      boolAttr(mm, "hdim_is_batch") || boolAttr(mm, "output_transpose") ||
      !module::getElementType(w.getOutput()).isF16() ||
      !module::getElementType(b.getOutput()).isF16())
    return false;
  auto x = mm->getOperand(0);
  auto y = mm.getOutput();
  auto dims = module::getShape(w.getOutput());
  if (dims.size() != 2)
    return false;
  auto k = dims[0], n = dims[1];
  if (!((k == 512 && n == 2048) || (k == 2048 && n == 512)) ||
      !hasShape(x, {1, 260, k}) || !hasShape(y, {1, 260, n}) ||
      !hasShape(b.getOutput(), {1, 1, n}))
    return false;
  auto weights = w.read<uint16_t>(), bias = b.read<uint16_t>();
  OpBuilder builder(mm);
  SmallVector<Value> halves;
  for (unsigned i = 0; i < 2; ++i) {
    auto weight = halfWeight(w, *weights, i, mm),
         biasPart = halfWeight(b, *bias, i, mm);
    SmallVector<Value> args(mm->getOperands());
    args[1] = weight;
    args[2] = biasPart;
    auto part = builder.create<tpu::MatMulOp>(
        suffix(builder, mm, i ? "split1" : "split0"),
        shaped(y, {1, 260, n / 2}), args, mm->getAttrs());
    halves.push_back(part.getOutput());
  }
  auto concat = builder.create<tpu::ConcatOp>(
      mm.getLoc(), y.getType(), halves,
      ArrayRef<NamedAttribute>{
          builder.getNamedAttr("axis", builder.getSI32IntegerAttr(2))});
  y.replaceAllUsesWith(concat.getOutput());
  mm.erase();
  w.erase();
  b.erase();
  return true;
}
bool specializeTpu(func::FuncOp func) {
  unsigned heads = 0, projections = 0, views = 0;
  for (auto &op : llvm::make_early_inc_range(func.getBody().front())) {
    if (auto mm = dyn_cast<tpu::MatMulOp>(op)) {
      auto name = module::getName(mm.getOperation());
      if (name.contains("/feed_forward/")) {
        if (!splitProjection(mm))
          return false;
        ++projections;
      } else if (name.contains("/self_attn/") &&
                 boolAttr(mm, "hdim_is_batch") &&
                 boolAttr(mm, "right_transpose") &&
                 !name.contains("/encoders0.0/")) {
        if (!headMajor(mm))
          return false;
        ++heads;
      }
    } else if (auto assign = dyn_cast<tpu::ShapeAssignOp>(op)) {
      if (!module::getName(assign.getOperation()).contains("/self_attn/"))
        continue;
      Value x = assign->getOperand(0), y = assign.getOutput();
      bool expand = hasShape(x, {1, 260, 512}) && hasShape(y, {1, 260, 4, 128});
      if (!expand &&
          !(hasShape(x, {1, 260, 4, 128}) && hasShape(y, {1, 260, 512})))
        continue; // Retain the mask ShapeAssign.
      OpBuilder b(assign);
      SmallVector<int64_t> dims = expand ? SmallVector<int64_t>{1, -1, 4, 128}
                                         : SmallVector<int64_t>{1, -1, 512};
      auto view =
          b.create<tpu::ReshapeOp>(assign.getLoc(), y.getType(), ValueRange{x},
                                   ArrayRef<NamedAttribute>{b.getNamedAttr(
                                       "shape", b.getI64ArrayAttr(dims))});
      y.replaceAllUsesWith(view.getOutput());
      assign.erase();
      ++views;
    }
  }
  return heads == 69 && projections == 140 && views == 280;
}

struct SenseVoiceUnpaddedPass
    : PassWrapper<SenseVoiceUnpaddedPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SenseVoiceUnpaddedPass)
  StringRef getArgument() const final { return "sensevoice-unpadded"; }
  StringRef getDescription() const final {
    return "Specialize the batch-one unpadded SenseVoice 2024-07-17 encoder";
  }
  void runOnOperation() override {
    auto m = getOperation();
    auto func = module::getMainFuncOp(m);
    if (!func || func.getNumArguments() != 4 || !module::isDynamic() ||
        !module::isCV84X6() || module::getCoreNum() != 1 ||
        module::getMode() != module::Mode::F16 ||
        !hasShape(func.getArgument(0), {1, 256, 560})) {
      m.emitError("expected CV84X6 FP16 single-core SenseVoice, [1,256,560]");
      signalPassFailure();
      return;
    }
    bool topStage = module::getState() == module::State::TOP_F32;
    if (!(topStage ? specializeTop(func) : specializeTpu(func))) {
      m.emitError("unexpected SenseVoice graph or mask; refusing unpadded "
                  "specialization");
      signalPassFailure();
    }
  }
};
} // namespace
void registerSenseVoiceUnpaddedPass() {
  PassRegistration<SenseVoiceUnpaddedPass>();
}
} // namespace tpu
} // namespace tpu_mlir
