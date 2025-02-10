//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToLinalg/OpLowering.h"

namespace tpu_mlir {

void populateTopToLinalgConversionPatterns(RewritePatternSet *patterns) {
  patterns->add<
      // clang-format off
        MaxPoolWithMaskLoweringToLinalg,
        AvgPoolLoweringToLinalg,
        ReduceLoweringToLinalg,
        BatchNormTrainLoweringToLinalg,
        LayerNormTrainLoweringToLinalg,
        InputLoweringToLinalg,
        TransposeLoweringToLinalg,
        ConvLoweringToLinalg,
        AddLoweringToLinalg,
        ReshapeLoweringToLinalg,
        SoftmaxLoweringToLinalg,
        PermuteLoweringToLinalg,
        SplitLoweringToLinalg,
        SliceLoweringToLinalg,
        MatMulLoweringToLinalg,
        VarianceLoweringToLinalg,
        UnsqueezeLoweringToLinalg,
        SqueezeLoweringToLinalg,
        // BroadcastLoweringToLinalg,
        AddConstLoweringToLinalg,
        DivLoweringToLinalg,
        RsqrtLoweringToLinalg,
        SubLoweringToLinalg,
        MulLoweringToLinalg,
        MulConstLoweringToLinalg,
        ExpLoweringToLinalg,
        ArgLoweringToLinalg
      // clang-format on
      >(patterns->getContext());
}

//===------------------------------------------------------------===//
// InputLowering
//===------------------------------------------------------------===//
void InputLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                     top::InputOp op) const {
  rewriter.replaceOp(op, op->getOperand(0));
}

int64_t toPositiveDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 ? dim : dim + inputRank;
}

bool isValidDim(int64_t dim, int64_t inputRank) {
  return dim >= 0 && dim < inputRank;
}

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
Value createInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                       Type elemTy, Value initElem) {
  Value initTensor =
      b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
  return b.create<linalg::FillOp>(loc, initElem, initTensor).getResult(0);
}

//===------------------------------------------------------------===//
// VarianceLoweringToLinalg
//===------------------------------------------------------------===//
void VarianceLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                        top::VarianceOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto input = op->getOperand(0);
  auto inputType = input.getType().cast<RankedTensorType>();

  llvm::DenseSet<int64_t> dimSet = {};
  bool keepDim = op.getKeepDims();
  for (auto dim : *module::getI64Array(op.getReduceList())) {
    dim = toPositiveDim(dim, inputType.getRank());
    // Drop invalid dimensions
    if (isValidDim(dim, inputType.getRank()))
      dimSet.insert(dim);
  }
  auto outElemType = outType.cast<RankedTensorType>().getElementType();

  auto reductionBodyBuilder = [&](OpBuilder &builder, Location loc,
                                  ValueRange payloadArgs) {
    Value result =
        rewriter.create<arith::AddFOp>(loc, payloadArgs[0], payloadArgs[1]);
    builder.create<linalg::YieldOp>(loc, result);
  };

  Value initElem = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(outElemType));

  auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  SmallVector<Value> resultShape;
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    auto currentDimSize = rewriter.create<tensor::DimOp>(loc, input, i);
    if (!dimSet.contains(i))
      resultShape.push_back(currentDimSize);
    else if (keepDim)
      resultShape.push_back(c1);
  }

  // Create the affine expressions that will be used to
  // iterate over the input and output tensors.
  // Here we also set the type of iterator: parallel or reduction.
  SmallVector<AffineExpr> exprs;
  SmallVector<utils::IteratorType> iteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  for (auto size : llvm::enumerate(inputType.getShape())) {
    exprs.push_back(rewriter.getAffineDimExpr(size.index()));

    if (dimSet.contains(size.index())) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
      // If `opInfo.keepDim`, create affine map to the first element
      // in the current dimension.
      if (keepDim)
        resultExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      iteratorTypes.push_back(utils::IteratorType::parallel);
      resultExprs.push_back(rewriter.getAffineDimExpr(size.index()));
    }
  }

  auto indexingMaps = AffineMap::inferFromExprList({exprs, resultExprs});
  Value accumulator = createInitTensor(rewriter, loc, resultShape,
                                       initElem.getType(), initElem);

  Value reduceOp = rewriter
                       .create<linalg::GenericOp>(
                           loc, /*resultTensorTypes=*/accumulator.getType(),
                           /*inputs=*/input,
                           /*outputs=*/accumulator, indexingMaps, iteratorTypes,
                           reductionBodyBuilder)
                       .getResult(0);

  rewriter.replaceOp(op, reduceOp);
}

Value createZeroInitTensor(OpBuilder &b, Location loc, ValueRange sizes,
                           Type elemTy) {
  Value initTensor =
      b.create<tensor::EmptyOp>(loc, getAsOpFoldResult(sizes), elemTy);
  RankedTensorType type = initTensor.getType().cast<RankedTensorType>();
  Value c0 =
      b.create<arith::ConstantOp>(loc, b.getZeroAttr(type.getElementType()));
  return b.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
}

//===------------------------------------------------------------===//
// ArgLoweringToLinalg
//===------------------------------------------------------------===//
void ArgLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                   top::ArgOp op) const {
  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  RankedTensorType idxResultType =
      op->getResult(1).getType().cast<RankedTensorType>();
  RankedTensorType inputType = input.getType().cast<RankedTensorType>();
  Type idxElementType = idxResultType.getElementType();

  bool keepDim = op.getKeepdims();
  int64_t dim = op.getAxis();
  dim = toPositiveDim(dim, inputType.getRank());
  if (!isValidDim(dim, inputType.getRank())) {
    llvm_unreachable("dim is not a valid dim\n");
  }

  Type inElementType = inputType.getElementType();
  // Constant op to account for the reduction along dim.
  auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
  SmallVector<Value> resultShape;
  auto inputShape = inputType.getShape();
  for (int64_t i = 0; i < inputType.getRank(); i++) {
    if (dim != i) {
      auto currentDimSize =
          rewriter.create<arith::ConstantIndexOp>(loc, inputShape[i]);
      resultShape.push_back(currentDimSize);
    } else if (keepDim)
      resultShape.push_back(c1);
  }
  // First fill the output buffer for the index.
  Value filledTensorIdx =
      createZeroInitTensor(rewriter, loc, resultShape, idxElementType);

  // Second fill the output buffer for the running max.
  Value initTensorMax = rewriter.create<tensor::EmptyOp>(
      loc, getAsOpFoldResult(resultShape), inElementType);

  Value fillValueMax = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getFloatAttr(inElementType, 0.0));

  Value filledTensorMax =
      rewriter.create<linalg::FillOp>(loc, fillValueMax, initTensorMax)
          .result();

  // Create the affine expressions that will be used to
  // iterate over the input and output tensors.
  // Here we also set the type of iterator: parallel or reduction.
  SmallVector<AffineExpr> exprs;
  SmallVector<utils::IteratorType> iteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  for (auto size : llvm::enumerate(inputType.getShape())) {
    exprs.push_back(rewriter.getAffineDimExpr(size.index()));

    if (unsigned(dim) == size.index()) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
      // If `keepDim`, create affine map to the first element
      // in the current dimension.
      if (keepDim)
        resultExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      iteratorTypes.push_back(utils::IteratorType::parallel);
      resultExprs.push_back(rewriter.getAffineDimExpr(size.index()));
    }
  }
  auto maps = AffineMap::inferFromExprList({exprs, resultExprs, resultExprs});
  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc,
      ArrayRef<Type>({filledTensorMax.getType(), filledTensorIdx.getType()}),
      input, ValueRange({filledTensorMax, filledTensorIdx}), maps,
      iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value newValue = blockArgs[0];
        Value oldValue = blockArgs[1];
        Value oldIndex = blockArgs[2];

        Value newIndex = rewriter.create<arith::IndexCastOp>(
            nestedLoc, rewriter.getI32Type(),
            rewriter.create<linalg::IndexOp>(loc, dim));
        Value newIndex2 = rewriter.create<arith::SIToFPOp>(
            nestedLoc, rewriter.getF32Type(), newIndex);

        Value resultMax =
            rewriter.create<arith::MaxFOp>(nestedLoc, newValue, oldValue);
        Value predicate = rewriter.create<arith::CmpFOp>(
            nestedLoc, arith::CmpFPredicate::OGT, newValue, oldValue);

        auto resultIndex = rewriter.create<arith::SelectOp>(
            nestedLoc, predicate, newIndex2, oldIndex);
        Value resultIndex2 = rewriter.create<arith::SIToFPOp>(
            nestedLoc, rewriter.getF32Type(), resultIndex);
        nestedBuilder.create<linalg::YieldOp>(
            nestedLoc, ValueRange({resultMax, resultIndex2}));
      });

  rewriter.replaceOp(op, linalgOp);
}

// //===------------------------------------------------------------===//
// // BroadcastLoweringToLinalg
// //===------------------------------------------------------------===//
// void BroadcastLoweringToLinalg::Lowering(PatternRewriter &rewriter,
// top::BroadcastOp op) const {
//   Location loc = op->getLoc();
//   auto outType = op->getResult(0).getType();
//   auto transpShape =
//   SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape()); Value
//   empty_bst = rewriter.create<tensor::EmptyOp>(loc, transpShape,
//   outType.cast<RankedTensorType>().getElementType());

//   auto t0 =
//   SmallVector<int64_t>(op->getOperand(0).getType().cast<RankedTensorType>().getShape());
//   for (auto i: t0) {
//     llvm::errs() <<"BroadcastOp getShape:"<<i<<"\n";
//   }

//   auto new_ops =
//       rewriter.create<linalg::BroadcastOp>(loc, op->getOperand(0), empty_bst,
//       llvm::ArrayRef<int64_t>{static_cast<long>(op.getDim())}).getResult();
//   for (auto new_op: new_ops) {
//       new_op.dump();
//       rewriter.replaceOp(op, new_op);
//       break;
//   }
// }

//===------------------------------------------------------------===//
// AddConstLoweringToLinalg
//===------------------------------------------------------------===//
void AddConstLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                        top::AddConstOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());

  auto constI32 = std::make_shared<std::vector<float>>(1, 0);
  constI32->data()[0] = op.getConstVal().convertToDouble();
  auto weight_type = RankedTensorType::get({1}, rewriter.getF32Type());
  auto c0 = top::WeightOp::create(op, "f32", *constI32, weight_type);

  // Value c0 = rewriter.create<arith::ConstantOp>(
  //     loc, FloatAttr::get(rewriter.getF32Type(),
  //     op.getConstVal().convertToDouble()));

  Value empty_bst = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  auto seq = llvm::to_vector<4>(llvm::seq<int64_t>(1, transpShape.size()));
  auto new_ops =
      rewriter.create<linalg::BroadcastOp>(loc, c0, empty_bst, seq).getResult();
  for (auto new_op : new_ops) {
    rewriter.replaceOp(op,
                       rewriter.create<linalg::AddOp>(
                           loc, ValueRange{op->getOperand(0), new_op}, empty));
    break;
  }
}

//===------------------------------------------------------------===//
// MulConstLoweringToLinalg
//===------------------------------------------------------------===//
void MulConstLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                        top::MulConstOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());

  auto constI32 = std::make_shared<std::vector<float>>(1, 0);
  constI32->data()[0] = op.getConstVal().convertToDouble();
  auto weight_type = RankedTensorType::get({1}, rewriter.getF32Type());
  auto c0 = top::WeightOp::create(op, "f32", *constI32, weight_type);

  // Value c0 = rewriter.create<arith::ConstantOp>(
  //     loc, FloatAttr::get(rewriter.getF32Type(),
  //     op.getConstVal().convertToDouble()));

  Value empty_bst = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  auto seq = llvm::to_vector<4>(llvm::seq<int64_t>(1, transpShape.size()));
  auto new_ops =
      rewriter.create<linalg::BroadcastOp>(loc, c0, empty_bst, seq).getResult();
  for (auto new_op : new_ops) {
    rewriter.replaceOp(op,
                       rewriter.create<linalg::MulOp>(
                           loc, ValueRange{op->getOperand(0), new_op}, empty));
    break;
  }
}

bool sameShape(SmallVector<int64_t> shapes1, SmallVector<int64_t> shapes2) {
  if (shapes1.size() != shapes2.size()) {
    return false;
  }

  for (int i = 0, sz = shapes1.size(); i < sz; i++) {
    if (shapes1[i] != shapes2[i]) {
      return false;
    }
  }

  return true;
}

//===------------------------------------------------------------===//
// DivLoweringToLinalg
//===------------------------------------------------------------===//
void DivLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                   top::DivOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto outShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, outShape, outType.cast<RankedTensorType>().getElementType());

  SmallVector<Value, 2> ins;
  for (int i = 0, sz = op->getOperands().size(); i < sz; i++) {
    auto in = op->getOperand(i);
    llvm::errs() << "wxc333333-"
                 << in.getType().cast<RankedTensorType>().getRank() << "---"
                 << outType.cast<RankedTensorType>().getRank() << "\n";
    if (!sameShape(SmallVector<int64_t>(
                       in.getType().cast<RankedTensorType>().getShape()),
                   outShape)) {
      SmallVector<Value> shapes;
      std::vector<int64_t> shapes2;
      for (int j = 0; j < outShape.size() - 1; j++) {
        shapes.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, outShape[j]));
        shapes2.push_back(outShape[j]);
      }
      // shapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
      // shapes2.push_back(1);

      auto toConcatIndexShape =
          rewriter.create<tensor::FromElementsOp>(loc, shapes);

      auto reshapeOp_type = RankedTensorType::get(ArrayRef<int64_t>{shapes2},
                                                  rewriter.getF32Type());
      auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
          loc, reshapeOp_type, in, toConcatIndexShape);

      Value empty_bst = rewriter.create<tensor::EmptyOp>(
          loc, outShape, outType.cast<RankedTensorType>().getElementType());
      auto BroadcastOps = rewriter
                              .create<linalg::BroadcastOp>(
                                  loc, reshapeOp, empty_bst,
                                  llvm::ArrayRef<int64_t>{
                                      static_cast<long>(outShape.size() - 1)})
                              .getResult();
      for (auto BroadcastOp : BroadcastOps) {
        ins.push_back(BroadcastOp);
        break;
      }
    } else {
      ins.push_back(in);
    }
  }
  rewriter.replaceOp(op, rewriter.create<linalg::DivOp>(loc, ins, empty));
  //提升其中1个操作数的rank，对增加的维度做broadcast
}

//===------------------------------------------------------------===//
// RsqrtLoweringToLinalg
//===------------------------------------------------------------===//
void RsqrtLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                     top::RsqrtOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  int inputRank = outType.cast<RankedTensorType>().getRank();
  auto input = op->getOperand(0);
  SmallVector<AffineMap> indexingMaps = {
      rewriter.getMultiDimIdentityMap(inputRank),
      rewriter.getMultiDimIdentityMap(inputRank)};
  SmallVector<utils::IteratorType> iteratorTypes(inputRank,
                                                 utils::IteratorType::parallel);
  Value rsqrt = rewriter
                    .create<linalg::GenericOp>(
                        loc, input.getType(), ValueRange{input}, input,
                        /*indexingMaps=*/indexingMaps,
                        /*iteratorTypes=*/iteratorTypes,
                        [&](OpBuilder &b, Location loc, ValueRange args) {
                          Value input = args[0];
                          Value result = b.create<math::SqrtOp>(loc, input);
                          b.create<linalg::YieldOp>(loc, result);
                        })
                    .getResult(0);
  rewriter.replaceOp(op, rsqrt);
}

//===------------------------------------------------------------===//
// SubLoweringToLinalg
//===------------------------------------------------------------===//
void SubLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                   top::SubOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());

  auto op1_shape =
      op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto op2_shape =
      op->getOperand(1).getType().cast<RankedTensorType>().getShape();
  if (op1_shape.size() != op2_shape.size()) {
    // auto is_op0_bst = op2_shape.size() > op1_shape.size();
    // auto op_bst = is_op0_bst?op->getOperand(0):op->getOperand(1);
    // Value empty_bst = rewriter.create<tensor::EmptyOp>(loc, transpShape,
    // outType.cast<RankedTensorType>().getElementType()); auto new_ops =
    //     rewriter.create<linalg::BroadcastOp>(loc, op_bst, empty_bst,
    //     llvm::ArrayRef<int64_t>{0}).getResult();
    // for (auto new_op: new_ops) {
    //   if (is_op0_bst) {
    //     rewriter.replaceOp(op, rewriter.create<linalg::AddOp>(loc,
    //     ValueRange{new_op, op->getOperand(1)}, empty));
    //   }
    //   else {
    //     rewriter.replaceOp(op, rewriter.create<linalg::AddOp>(loc,
    //     ValueRange{op->getOperand(0), new_op}, empty));
    //   }
    //   break;
    // }
  } else {
    // int i = 0;
    // for (const auto &[dim1, dim2] : llvm::zip(op1_shape, op2_shape)) {
    //   if (dim1 != dim2) {
    //     assert(dim1 == 1 || dim2 == 1);
    //     if (dim1 == 1)
    //   }
    //   i++;
    // }

    SmallVector<Value> shapes;
    std::vector<int64_t> shapes2;
    for (int i = 0; i < op2_shape.size() - 1; i++) {
      shapes.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, op2_shape[i]));
      shapes2.push_back(op2_shape[i]);
    }
    auto toConcatIndexShape =
        rewriter.create<tensor::FromElementsOp>(loc, shapes);

    auto reshapeOp_type = RankedTensorType::get(ArrayRef<int64_t>{shapes2},
                                                rewriter.getF32Type());
    auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
        loc, reshapeOp_type, op->getOperand(1), toConcatIndexShape);

    Value empty_bst = rewriter.create<tensor::EmptyOp>(
        loc, transpShape, outType.cast<RankedTensorType>().getElementType());
    auto BroadcastOps = rewriter
                            .create<linalg::BroadcastOp>(
                                loc, reshapeOp, empty_bst,
                                llvm::ArrayRef<int64_t>{
                                    static_cast<long>(op2_shape.size() - 1)})
                            .getResult();
    for (auto BroadcastOp : BroadcastOps) {
      rewriter.replaceOp(
          op, rewriter.create<linalg::SubOp>(
                  loc, ValueRange{op->getOperand(0), BroadcastOp}, empty));
      break;
    }
  }
}

//===------------------------------------------------------------===//
// MulLoweringToLinalg
//===------------------------------------------------------------===//
void MulLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                   top::MulOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());

  auto op1_shape =
      op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto op2_shape =
      op->getOperand(1).getType().cast<RankedTensorType>().getShape();
  if (op1_shape.size() != op2_shape.size()) {
    Value empty_bst = rewriter.create<tensor::EmptyOp>(
        loc, transpShape, outType.cast<RankedTensorType>().getElementType());
    auto BroadcastOps =
        rewriter
            .create<linalg::BroadcastOp>(loc, op->getOperand(1), empty_bst,
                                         llvm::ArrayRef<int64_t>{0, 1})
            .getResult();
    for (auto BroadcastOp : BroadcastOps) {
      rewriter.replaceOp(
          op, rewriter.create<linalg::MulOp>(
                  loc, ValueRange{op->getOperand(0), BroadcastOp}, empty));
      break;
    }
  } else {
    // int i = 0;
    // for (const auto &[dim1, dim2] : llvm::zip(op1_shape, op2_shape)) {
    //   if (dim1 != dim2) {
    //     assert(dim1 == 1 || dim2 == 1);
    //     if (dim1 == 1)
    //   }
    //   i++;
    // }

    SmallVector<Value> shapes;
    std::vector<int64_t> shapes2;
    for (int i = 0; i < op2_shape.size() - 1; i++) {
      shapes.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, op2_shape[i]));
      shapes2.push_back(op2_shape[i]);
    }
    auto toConcatIndexShape =
        rewriter.create<tensor::FromElementsOp>(loc, shapes);

    auto reshapeOp_type = RankedTensorType::get(ArrayRef<int64_t>{shapes2},
                                                rewriter.getF32Type());
    auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
        loc, reshapeOp_type, op->getOperand(1), toConcatIndexShape);

    Value empty_bst = rewriter.create<tensor::EmptyOp>(
        loc, transpShape, outType.cast<RankedTensorType>().getElementType());
    auto BroadcastOps = rewriter
                            .create<linalg::BroadcastOp>(
                                loc, reshapeOp, empty_bst,
                                llvm::ArrayRef<int64_t>{
                                    static_cast<long>(op2_shape.size() - 1)})
                            .getResult();
    for (auto BroadcastOp : BroadcastOps) {
      rewriter.replaceOp(
          op, rewriter.create<linalg::MulOp>(
                  loc, ValueRange{op->getOperand(0), BroadcastOp}, empty));
      break;
    }
  }
}

//===------------------------------------------------------------===//
// ExpLoweringToLinalg
//===------------------------------------------------------------===//
void ExpLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                   top::ExpOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  rewriter.replaceOp(op, rewriter.create<linalg::ExpOp>(
                             loc, ValueRange{op->getOperand(0)}, empty));
}

//===------------------------------------------------------------===//
// AddLoweringToLinalg
//===------------------------------------------------------------===//
void AddLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                   top::AddOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  auto op1_shape =
      op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  auto op2_shape =
      op->getOperand(1).getType().cast<RankedTensorType>().getShape();
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  if (op1_shape.size() != op2_shape.size()) {
    // auto max_dim = max(op1_shape.size() ,op2_shape.size());
    // for (int i = 0; i < max_dim; i++) {

    // }
    auto is_op0_bst = op2_shape.size() > op1_shape.size();
    int si = is_op0_bst ? op2_shape.size() - op1_shape.size()
                        : op1_shape.size() - op2_shape.size();
    auto seq = llvm::to_vector<4>(llvm::seq<int64_t>(0, si));
    auto op_bst = is_op0_bst ? op->getOperand(0) : op->getOperand(1);
    Value empty_bst = rewriter.create<tensor::EmptyOp>(
        loc, transpShape, outType.cast<RankedTensorType>().getElementType());
    auto new_ops =
        rewriter.create<linalg::BroadcastOp>(loc, op_bst, empty_bst, seq)
            .getResult();
    for (auto new_op : new_ops) {
      if (is_op0_bst) {
        rewriter.replaceOp(
            op, rewriter.create<linalg::AddOp>(
                    loc, ValueRange{new_op, op->getOperand(1)}, empty));
      } else {
        rewriter.replaceOp(
            op, rewriter.create<linalg::AddOp>(
                    loc, ValueRange{op->getOperand(0), new_op}, empty));
      }
      break;
    }
  } else {
    rewriter.replaceOp(
        op, rewriter.create<linalg::AddOp>(
                loc, ValueRange{op->getOperand(0), op->getOperand(1)}, empty));
  }
}

//===------------------------------------------------------------===//
// ReshapeLoweringToLinalg
//===------------------------------------------------------------===//
void ReshapeLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                       top::ReshapeOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto shape = outType.cast<RankedTensorType>().getShape();
  if (op.getShape().has_value()) {
    auto shape = module::getI64Array(op.getShape().value());
  }
  SmallVector<Value> shapes;
  for (int i = 0; i < shape.size(); i++) {
    shapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, shape[i]));
  }
  auto toConcatIndexShape =
      rewriter.create<tensor::FromElementsOp>(loc, shapes);
  auto new_op = rewriter.create<tensor::ReshapeOp>(
      loc, outType, op->getOperand(0), toConcatIndexShape);
  rewriter.replaceOp(op, new_op);
}

//===------------------------------------------------------------===//
// SqueezeLoweringToLinalg
//===------------------------------------------------------------===//
void SqueezeLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                       top::SqueezeOp op) const {
  //   Location loc = op->getLoc();
  //   auto outType = op->getResult(0).getType();
  //   auto shape = outType.cast<RankedTensorType>().getShape();
  //   if (op.getShape().has_value()) {
  //     auto shape = module::getI64Array(op.getShape().value());
  //   }
  //   auto axes = op.getAxes();
  //   SmallVector<Value> shapes;
  //   for (int i = 0; i < shape.size(); i++) {
  //     shapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc,
  //     shape[i]));
  //   }
  //   auto toConcatIndexShape =
  //       rewriter.create<tensor::FromElementsOp>(loc, shapes);
  //   auto new_op =
  //       rewriter.create<tensor::ReshapeOp>(loc, outType, op->getOperand(0),
  //       toConcatIndexShape);
  //   rewriter.replaceOp(op, new_op);
}

// DenseSet<int64_t> dimensionsToReduce;
//===------------------------------------------------------------===//
// UnsqueezeLoweringToLinalg
//===------------------------------------------------------------===//
void UnsqueezeLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                         top::UnsqueezeOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto inShape = SmallVector<int64_t>(
      op->getOperand(0).getType().cast<RankedTensorType>().getShape());
  auto axes = op.getAxes().getAsValueRange<IntegerAttr>();
  SmallVector<Value> shapes;
  for (int i = 0, j = 0; i < op.getAxes().size() + inShape.size(); i++) {
    if (!llvm::count(axes, i))
      shapes.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, inShape[j++]));
    else
      shapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }
  auto target_shape = rewriter.create<tensor::FromElementsOp>(loc, shapes);
  auto new_op = rewriter.create<tensor::ReshapeOp>(
      loc, outType, op->getOperand(0), target_shape);
  rewriter.replaceOp(op, new_op);
}

// void UnsqueezeLoweringToLinalg::Lowering(PatternRewriter &rewriter,
// top::UnsqueezeOp op) const {
//   Location loc = op->getLoc();
//   auto outType = op->getResult(0).getType();
//   auto inShape =
//   SmallVector<int64_t>(op->getOperand(0).getType().cast<RankedTensorType>().getShape());
//   auto axes = *module::getI64Array(op.getAxes());
//   SmallVector<Value> shapes;
//   for (int i = 0, j = 0; i < axes.size() + inShape.size(); i++) {
//     if (!std::count(axes.begin(), axes.end(), i)) {
//       shapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc,
//       inShape[j++]));
//     }
//     else
//       shapes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
//   }
//   auto toConcatIndexShape =
//       rewriter.create<tensor::FromElementsOp>(loc, shapes);
//   auto new_op =
//       rewriter.create<tensor::ReshapeOp>(loc, outType, op->getOperand(0),
//       toConcatIndexShape);
//   rewriter.replaceOp(op, new_op);
// }

//===------------------------------------------------------------===//
// SoftmaxLoweringToLinalg
//===------------------------------------------------------------===//
void SoftmaxLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                       top::SoftmaxOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  auto new_op = rewriter.create<linalg::SoftmaxOp>(
      loc, outType, op->getOperand(0), empty,
      IntegerAttr::get(rewriter.getI64Type(), op.getAxis()));
  rewriter.replaceOp(op, new_op);
}

//===------------------------------------------------------------===//
// SplitLoweringToLinalg
//===------------------------------------------------------------===//
void SplitLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                     top::SplitOp op) const {
  Location loc = op->getLoc();
  auto input = op->getOperand(0);
  RankedTensorType inputType =
      input.getType().template cast<RankedTensorType>();
  auto axis = op.getAxis();
  auto num = op.getNum();
  assert(inputType.getShape()[axis] % num == 0);

  auto outType = op->getResult(0).getType();
  auto resultShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  auto slice_len = resultShape[axis];

  SmallVector<Value> offsets;
  SmallVector<Value> strides;
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  strides.resize(inputType.getRank(), one);
  offsets.resize(inputType.getRank(), zero);
  SmallVector<Value> resultShapes;
  for (int i = 0; i < resultShape.size(); i++) {
    resultShapes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
  }

  SmallVector<Value> results;
  for (int i = 0; i < num; i++) {
    offsets[axis] = rewriter.create<arith::ConstantIndexOp>(loc, i * slice_len);
    Value result = rewriter.create<tensor::ExtractSliceOp>(
        loc, input, offsets, resultShapes, strides);
    results.push_back(result);
  }
  rewriter.replaceOp(op, results);
}

//===------------------------------------------------------------===//
// SliceLoweringToLinalg
//===------------------------------------------------------------===//
void SliceLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                     top::SliceOp op) const {
  Location loc = op->getLoc();
  auto input = op->getOperand(0);
  RankedTensorType inputType =
      input.getType().template cast<RankedTensorType>();
  // auto axes = op.getAxes();
  auto inShape = inputType.getShape();

  auto outType = op->getResult(0).getType();
  auto resultShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());

  SmallVector<Value> offsets;
  SmallVector<Value> ends;
  SmallVector<Value> steps;
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  steps.resize(inputType.getRank(), one);
  offsets.resize(inputType.getRank(), zero);
  ends.resize(inputType.getRank(), zero);
  SmallVector<Value> resultShapes;
  for (int i = 0; i < resultShape.size(); i++) {
    resultShapes.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, resultShape[i]));
  }

  for (const auto &en : llvm::enumerate(op.getOffset())) {
    int tmp =
        en.value().dyn_cast<mlir::IntegerAttr>().getValue().getZExtValue();
    if (tmp != 0)
      offsets[en.index()] = rewriter.create<arith::ConstantIndexOp>(loc, tmp);
  }

  for (const auto &en : llvm::enumerate(op.getSteps())) {
    int tmp =
        en.value().dyn_cast<mlir::IntegerAttr>().getValue().getZExtValue();
    if (tmp != 1)
      steps[en.index()] = rewriter.create<arith::ConstantIndexOp>(loc, tmp);
  }

  for (const auto &en : llvm::enumerate(op.getEnds())) {
    int tmp =
        en.value().dyn_cast<mlir::IntegerAttr>().getValue().getZExtValue();
    if (tmp < 0)
      tmp += inShape[en.index()];
    ends[en.index()] = rewriter.create<arith::ConstantIndexOp>(loc, tmp);
  }

  Value result = rewriter.create<tensor::ExtractSliceOp>(loc, input, offsets,
                                                         resultShapes, steps);
  rewriter.replaceOp(op, result);
}

//===------------------------------------------------------------===//
// ConvLoweringToLinalg
//===------------------------------------------------------------===//
void ConvLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                    top::ConvOp op) const {
  auto strides = module::getI64Array(op.getStrides());
  auto dilations = module::getI64Array(op.getDilations(), 2, 1);
  Location loc = op->getLoc();
  auto input = op->getOperand(0);
  auto filter = op->getOperand(1);
  auto outType = op->getResult(0).getType();
  auto outShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());

  Value initTensor =
      rewriter.create<tensor::EmptyOp>(loc, outShape, rewriter.getF32Type());
  Value c0 = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(rewriter.getF32Type(), 0.0));
  Value zeroFill =
      rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
  Value conv2d = rewriter
                     .create<linalg::Conv2DNchwFchwOp>(
                         loc, zeroFill.getType(), ValueRange{input, filter},
                         zeroFill, rewriter.getDenseI64ArrayAttr(*strides),
                         rewriter.getDenseI64ArrayAttr(*dilations))
                     .getResult(0);
  rewriter.replaceOp(op, conv2d);
}

/// Inverted STD: rSTD = 1 / sqrt(var + eps).
Value calculateRSTD(OpBuilder &b, Location loc, Type elemTy, Value eps,
                    Value var) {
  // The eps is always f64.
  Value truncatedEps = b.create<arith::TruncFOp>(loc, elemTy, eps);
  Value varPlusEps = b.create<arith::AddFOp>(loc, var, truncatedEps);
  Value rSTD = b.create<math::RsqrtOp>(loc, varPlusEps);
  return rSTD;
}

// Normalization formula:
//   ((input - mean) * rSTD * weight + bias
Value createLinalgPayloadCalculationForNormOpsWithRSTD(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean,
    Value rSTD, Value eps, Value weight, Value bias) {
  Value inputSubMean = b.create<arith::SubFOp>(loc, input, mean);
  Value temp = b.create<arith::MulFOp>(loc, inputSubMean, rSTD);
  Value timesWeight = b.create<arith::MulFOp>(loc, temp, weight);
  Value plusBias = b.create<arith::AddFOp>(loc, timesWeight, bias);
  return plusBias;
}

Value createLinalgPayloadCalculationForNormOpsWithVar(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean, Value var,
    Value eps, Value weight, Value bias) {
  Value rSTD = calculateRSTD(b, loc, elemTy, eps, var);
  Value result = createLinalgPayloadCalculationForNormOpsWithRSTD(
      b, loc, elemTy, input, mean, rSTD, eps, weight, bias);
  return result;
}

//===------------------------------------------------------------===//
// BatchNormTrainLoweringToLinalg
//===------------------------------------------------------------===//
void BatchNormTrainLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                              top::BatchNormTrainOp op) const {
  // Location loc = op->getLoc();
  // MLIRContext *context = op->getContext();
  // auto input = op.getInput();
  // auto weight = op.getGamma();
  // auto bias = op.getBeta();
  // auto runningMean = op.getMean();
  // auto runningVar = op.getVariance();
  // auto inputType = input.getType().cast<RankedTensorType>();
  // auto weightType = weight.getType().cast<RankedTensorType>();
  // auto biasType = bias.getType().cast<RankedTensorType>();
  // auto runningMeanType = runningMean.getType().cast<RankedTensorType>();
  // auto runningVarType = runningVar.getType().cast<RankedTensorType>();
  // Value eps = rewriter.create<arith::ConstantOp>(
  //     loc,
  //     FloatAttr::get(rewriter.getF32Type(),
  //     op.getEpsilon().convertToDouble()));

  // auto inputRank = inputType.getRank();
  // if (inputRank < 2)
  //   llvm_unreachable("input should have rank larger than 1");

  // if (weightType.getRank() != 1 || biasType.getRank() != 1 ||
  //     runningMeanType.getRank() != 1 || runningVarType.getRank() != 1) {
  //   llvm_unreachable(
  //       "expect weight, bias, running_mean and running_var to be rank 1");
  // }

  // auto indexingMap = AffineMap::get(
  //     /*dimCount=*/inputRank,
  //     /*symbolCount=*/0, rewriter.getAffineDimExpr(1), context);
  // SmallVector<AffineMap> indexingMaps = {
  //     rewriter.getMultiDimIdentityMap(inputRank), // input
  //     indexingMap,                                // weight
  //     indexingMap,                                // bias
  //     indexingMap,                                // runningMean
  //     indexingMap,                                // runningVar
  //     rewriter.getMultiDimIdentityMap(inputRank), // output
  // };
  // SmallVector<utils::IteratorType> iteratorTypes(inputRank,
  //                                                utils::IteratorType::parallel);
  // Value batchNorm =
  //     rewriter
  //         .create<linalg::GenericOp>(
  //             loc, input.getType(),
  //             ValueRange{input, weight, bias, runningMean, runningVar},
  //             input,
  //             /*indexingMaps=*/indexingMaps,
  //             /*iteratorTypes=*/iteratorTypes,
  //             [&](OpBuilder &b, Location loc, ValueRange args) {
  //               Value input = args[0], weight = args[1], bias = args[2],
  //                     mean = args[3], var = args[4];
  //               Value result =
  //               createLinalgPayloadCalculationForNormOpsWithVar(
  //                   b, loc, var.getType(), input, mean, var, eps, weight,
  //                   bias);
  //               b.create<linalg::YieldOp>(loc, result);
  //             })
  //         .getResult(0);
  // rewriter.replaceOp(op, {batchNorm, runningMean, runningVar});
}

//===------------------------------------------------------------===//
// LayerNormTrainLoweringToLinalg
//===------------------------------------------------------------===//
void LayerNormTrainLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                              top::LayerNormTrainOp op) const {
  // Location loc = op->getLoc();
  // MLIRContext *context = op->getContext();
  // auto input = op.getInput();
  // auto weight = op.getGamma();
  // auto bias = op.getBeta();
  // auto runningMean = op.getMean();
  // auto runningVar = op.getVariance();
  // auto inputType = input.getType().cast<RankedTensorType>();
  // auto weightType = weight.getType().cast<RankedTensorType>();
  // auto biasType = bias.getType().cast<RankedTensorType>();
  // auto runningMeanType = runningMean.getType().cast<RankedTensorType>();
  // auto runningVarType = runningVar.getType().cast<RankedTensorType>();
  // Value eps = rewriter.create<arith::ConstantOp>(
  //     loc, FloatAttr::get(rewriter.getF32Type(),
  //     op.getEpsilon().convertToDouble()));

  // auto inputRank = inputType.getRank();
  // if (inputRank < 2)
  //   llvm_unreachable("input should have rank larger than 1");

  // if (weightType.getRank() != 1 || biasType.getRank() != 1 ||
  //     runningMeanType.getRank() != 1 || runningVarType.getRank() != 1) {
  //   llvm_unreachable("expect weight, bias, running_mean and running_var to be
  //   rank 1");
  // }

  // auto indexingMap = AffineMap::get(
  //     /*dimCount=*/inputRank,
  //     /*symbolCount=*/0, rewriter.getAffineDimExpr(1), context);
  // SmallVector<AffineMap> indexingMaps = {
  //     rewriter.getMultiDimIdentityMap(inputRank), // input
  //     indexingMap,                                // weight
  //     indexingMap,                                // bias
  //     indexingMap,                                // runningMean
  //     indexingMap,                                // runningVar
  //     rewriter.getMultiDimIdentityMap(inputRank), // output
  // };
  // SmallVector<utils::IteratorType> iteratorTypes(
  //     inputRank, utils::IteratorType::parallel);
  // Value batchNorm =
  //     rewriter
  //         .create<linalg::GenericOp>(
  //             loc, input.getType(),
  //             ValueRange{input, weight, bias, runningMean, runningVar},
  //             input,
  //             /*indexingMaps=*/indexingMaps,
  //             /*iteratorTypes=*/iteratorTypes,
  //             [&](OpBuilder &b, Location loc, ValueRange args) {
  //               Value input = args[0], weight = args[1], bias = args[2],
  //                     mean = args[3], var = args[4];
  //               Value result =
  //                   createLinalgPayloadCalculationForNormOpsWithVar(
  //                       b, loc, var.getType(), input, mean, var, eps, weight,
  //                       bias);
  //               b.create<linalg::YieldOp>(loc, result);
  //             })
  //         .getResult(0);
  // rewriter.replaceOp(op, {batchNorm, runningMean, runningVar});
}

//===------------------------------------------------------------===//
// AvgPoolLoweringToLinalg
//===------------------------------------------------------------===//
void AvgPoolLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                       top::AvgPoolOp op) const {}

Value castIndexToInt64(OpBuilder &b, Location loc, Value idx) {
  assert(idx.getType().isa<IndexType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
}

SmallVector<Value> getAsConstantIndexValues(OpBuilder &b, Location loc,
                                            SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<arith::ConstantOp>(loc, b.getIndexAttr(val));
  }));
}

// Creates a tensor with required `sizes` and `elemTy` and fills it with
// initElem.
Value createInitTensor(OpBuilder &b, Location loc, SmallVector<int64_t> sizes,
                       Type elemTy, Value initElem) {
  Value initTensor = b.create<tensor::EmptyOp>(loc, sizes, elemTy);
  return b.create<linalg::FillOp>(loc, initElem, initTensor).getResult(0);
}

static SmallVector<OpFoldResult>
getIndexIntsAsOpFoldResult(OpBuilder &b, SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(
      ints, [&](int64_t val) -> OpFoldResult { return b.getIndexAttr(val); }));
}

// Helper function to get the padding tensor given the padding int values.
Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                      SmallVectorImpl<int64_t> &lowPaddingInts,
                      SmallVectorImpl<int64_t> &highPaddingInts, Value pad) {
  Location loc = op->getLoc();
  Type rankedTensorType =
      tensor::PadOp::inferResultType(input.getType().cast<RankedTensorType>(),
                                     lowPaddingInts, highPaddingInts);
  SmallVector<OpFoldResult> lowPaddings =
      getIndexIntsAsOpFoldResult(b, lowPaddingInts);
  SmallVector<OpFoldResult> highPaddings =
      getIndexIntsAsOpFoldResult(b, highPaddingInts);
  Value paddedInput =
      b.create<tensor::PadOp>(loc, rankedTensorType, input, /*low=*/lowPaddings,
                              /*high=*/highPaddings, pad);
  return paddedInput;
}

//===------------------------------------------------------------===//
// MaxPoolWithMaskLoweringToLinalg
//===------------------------------------------------------------===//
void MaxPoolWithMaskLoweringToLinalg::Lowering(
    PatternRewriter &rewriter, top::MaxPoolWithMaskOp op) const {
  auto strideInts = module::getI64Array(op.getStrides());
  auto paddingInts = module::getI64Array(op.getPads());
  auto dilationInts = std::make_shared<std::vector<int64_t>>(2, 1);
  Location loc = op->getLoc();
  auto input = op->getOperand(0);
  auto outType = op->getResult(0).getType();
  auto outShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());

  Value initTensor =
      rewriter.create<tensor::EmptyOp>(loc, outShape, rewriter.getF32Type());
  Value c0 = rewriter.create<arith::ConstantOp>(
      loc, FloatAttr::get(rewriter.getF32Type(), 0.0));
  Value zeroFill =
      rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);

  auto elementType = outType.cast<RankedTensorType>().getElementType();

  SmallVector<int64_t> tmp0;
  auto t0 = module::getI64Array(op.getKernelShape());
  llvm::errs() << "tmp0:" << op.getKernelShape() << ", " << t0->at(0) << "\n";
  for (auto i : *t0) {
    tmp0.push_back(i);
    llvm::errs() << "tmp0:" << i << "\n";
  }

  Value windowTensor = rewriter.create<tensor::EmptyOp>(loc, tmp0, elementType);

  SmallVector<int64_t> tmp, tmp2, tmp3;
  for (auto i : *paddingInts) {
    tmp.push_back(i);
  }
  for (auto i : *dilationInts) {
    tmp2.push_back(i);
  }
  for (auto i : *strideInts) {
    tmp3.push_back(i);
  }
  SmallVector<Value> padding = getAsConstantIndexValues(rewriter, loc, tmp);
  SmallVector<Value> dilation = getAsConstantIndexValues(rewriter, loc, tmp2);
  SmallVector<Value> stride = getAsConstantIndexValues(rewriter, loc, tmp3);

  SmallVector<int64_t> lowPaddingIncludingNC = {0, 0};
  lowPaddingIncludingNC.append(tmp);
  SmallVector<int64_t> highPaddingIncludingNC = lowPaddingIncludingNC;

  int dimensionality = 2;
  if (op.getCeilMode().has_value() && op.getCeilMode().value()) {
    for (int64_t i = 0; i < dimensionality; ++i) {
      highPaddingIncludingNC[i + 2] += tmp3[i];
    }
  }

  auto initValueAttr = rewriter.getFloatAttr(
      elementType,
      APFloat::getInf(elementType.cast<mlir::FloatType>().getFloatSemantics(),
                      /*Negative=*/true));

  Value initValue =
      rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(initValueAttr));
  Value paddedInput =
      getPaddedTensor(op.getOperation(), rewriter, input, lowPaddingIncludingNC,
                      highPaddingIncludingNC, initValue);

  Value MaxPool =
      rewriter
          .create<linalg::PoolingNchwMaxOp>(
              loc, zeroFill.getType(), ValueRange{paddedInput, windowTensor},
              zeroFill, rewriter.getDenseI64ArrayAttr(*strideInts),
              rewriter.getDenseI64ArrayAttr(*dilationInts))
          .getResult(0);

  Value cstMinusOne =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(-1));
  auto indiceselementType =
      op->getResult(1).getType().cast<RankedTensorType>().getElementType();
  Value indicesTensor = createInitTensor(rewriter, loc, outShape,
                                         indiceselementType, cstMinusOne);

  SmallVector<AffineExpr> inputExprs, outputExprs, kernelExprs;
  for (unsigned i = 0; i < 4; i++) {
    inputExprs.push_back(rewriter.getAffineDimExpr(i));
    outputExprs.push_back(rewriter.getAffineDimExpr(i));
  }
  kernelExprs.push_back(rewriter.getAffineDimExpr(4));
  kernelExprs.push_back(rewriter.getAffineDimExpr(5));

  // Here we have six dimensions, each corresponding to N, C, Hout, Wout, kH,
  // and kW, respectively, as described in the algorithm above.
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList({inputExprs, kernelExprs, outputExprs});
  SmallVector<utils::IteratorType> iteratorTypes(4,
                                                 utils::IteratorType::parallel);
  iteratorTypes.push_back(utils::IteratorType::reduction);
  iteratorTypes.push_back(utils::IteratorType::reduction);

  Value inputShapeW = rewriter.create<arith::ConstantOp>(
      loc, IntegerAttr::get(rewriter.getI32Type(), outShape[3]));
  // Input format is : [N, C, H, W]
  // Value inputShapeW = getDimOp(rewriter, loc, input, 3);

  Value indicesResult =
      rewriter
          .create<linalg::GenericOp>(
              loc, /*resultTensorTypes=*/indicesTensor.getType(),
              /*inputs=*/ValueRange({MaxPool, windowTensor}),
              /*outputs=*/indicesTensor,
              /*indexingMaps=*/indexingMaps,
              /*iteratorTypes=*/iteratorTypes,
              [&](OpBuilder &b, Location loc, ValueRange args) {
                Value maxVal = args[0], res = args[2];

                Value i = b.create<linalg::IndexOp>(loc, 0);
                Value j = b.create<linalg::IndexOp>(loc, 1);
                Value m = b.create<linalg::IndexOp>(loc, 2);
                Value n = b.create<linalg::IndexOp>(loc, 3);
                Value p = b.create<linalg::IndexOp>(loc, 4);
                Value r = b.create<linalg::IndexOp>(loc, 5);

                Value mTimesStride = b.create<arith::MulIOp>(loc, m, stride[0]);
                Value pTimesDilation =
                    b.create<arith::MulIOp>(loc, p, dilation[0]);
                Value indexH =
                    b.create<arith::AddIOp>(loc, mTimesStride, pTimesDilation);
                Value nTimesStride = b.create<arith::MulIOp>(loc, n, stride[1]);
                Value rTimesDilation =
                    b.create<arith::MulIOp>(loc, r, dilation[1]);
                Value indexW =
                    b.create<arith::AddIOp>(loc, nTimesStride, rTimesDilation);
                Value input = b.create<tensor::ExtractOp>(
                    loc, paddedInput, ValueRange{i, j, indexH, indexW});
                Value pred = b.create<arith::CmpFOp>(
                    loc, arith::CmpFPredicate::OEQ, input, maxVal);

                Value indexHMinusPadding =
                    b.create<arith::SubIOp>(loc, indexH, padding[0]);
                Value indexWMinusPadding =
                    b.create<arith::SubIOp>(loc, indexW, padding[1]);
                Value outIndex = b.create<arith::MulIOp>(
                    loc, indexHMinusPadding, inputShapeW);
                outIndex =
                    b.create<arith::AddIOp>(loc, outIndex, indexWMinusPadding);
                Value result = b.create<arith::SelectOp>(
                    loc, pred, castIndexToInt64(b, loc, outIndex), res);

                Value predInvalidIndex = b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::eq, res, cstMinusOne);
                Value out = b.create<arith::SelectOp>(loc, predInvalidIndex,
                                                      result, res);

                b.create<linalg::YieldOp>(loc, out);
              })
          .getResult(0);

  rewriter.replaceOp(op, {MaxPool, indicesResult});
}

//===------------------------------------------------------------===//
// TransposeLoweringToLinalg
//===------------------------------------------------------------===//
void TransposeLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                         top::TransposeOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  auto shape_size = transpShape.size();
  auto dim0_ = op.getDim0();
  if (dim0_ < 0)
    dim0_ += shape_size;
  auto dim1_ = op.getDim1();
  if (dim1_ < 0)
    dim1_ += shape_size;
  SmallVector<int64_t> perm(shape_size);
  for (int i = 0; i < shape_size; i++) {
    perm[i] = i;
  }
  int tmp = perm[dim0_];
  perm[dim0_] = perm[dim1_];
  perm[dim1_] = tmp;
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  auto ts_op = rewriter.create<linalg::TransposeOp>(
      loc, op->getOperand(0), empty, rewriter.getDenseI64ArrayAttr(perm));
  rewriter.replaceOp(op, ts_op);
}

//===------------------------------------------------------------===//
// PermuteLoweringToLinalg
//===------------------------------------------------------------===//
void PermuteLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                       top::PermuteOp op) const {
  Location loc = op->getLoc();
  auto outType = op->getResult(0).getType();
  auto transpShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, transpShape, outType.cast<RankedTensorType>().getElementType());
  auto ts_op = rewriter.create<linalg::TransposeOp>(
      loc, op->getOperand(0), empty,
      rewriter.getDenseI64ArrayAttr(*module::getI64Array(op.getOrder())));
  rewriter.replaceOp(op, ts_op);
}

//===------------------------------------------------------------===//
// ReduceLoweringToLinalg
//===------------------------------------------------------------===//
void ReduceLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                      top::ReduceOp op) const {
  Location loc = op->getLoc();
  auto inType = op->getOperand(0).getType();
  auto inShape = inType.cast<RankedTensorType>().getShape();

  auto axes = module::getI64Array(op.getAxes());
  std::vector<int64_t> shapes2;
  for (auto i : *axes) {
    if (i < 0) {
      i += inShape.size();
    }
    shapes2.push_back(i);
    // llvm::errs() <<"Axes:"<<i<<"\n";
  }

  SmallVector<int64_t> reducedInputDims;
  for (const auto &en : llvm::enumerate(inShape)) {
    if (!std::count(shapes2.begin(), shapes2.end(), en.index()))
      reducedInputDims.push_back(en.value());
  }

  Value empty = rewriter.create<tensor::EmptyOp>(
      loc, reducedInputDims, inType.cast<RankedTensorType>().getElementType());
  auto tmp_op = rewriter.create<linalg::ReduceOp>(
      loc, ValueRange{op->getOperand(0)}, ValueRange{empty}, shapes2,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value in = args[0], init = args[1];
        Value res = b.create<arith::AddFOp>(loc, init, in);
        b.create<linalg::YieldOp>(loc, res);
      });
  tmp_op.dump();
  rewriter.replaceOp(op, tmp_op);
}

//===------------------------------------------------------------===//
// MatMulLowering
//===------------------------------------------------------------===//
void MatMulLoweringToLinalg::Lowering(PatternRewriter &rewriter,
                                      top::MatMulOp op) const {
  Location loc = op->getLoc();
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  auto lhs_type = lhs.getType().cast<RankedTensorType>();
  auto rhs_type = rhs.getType().cast<RankedTensorType>();
  auto lhs_rank = lhs_type.getRank();
  auto rhs_rank = rhs_type.getRank();
  auto outType = op->getResult(0).getType();
  auto outShape =
      SmallVector<int64_t>(outType.cast<RankedTensorType>().getShape());
  if (lhs_rank == 2 && rhs_rank == 2) {
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, outShape, rewriter.getF32Type());
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(rewriter.getF32Type(), 0.0));
    Value zeroFill =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                                 ValueRange{lhs, rhs}, zeroFill)
                       .getResult(0);
    rewriter.replaceOp(op, matmul);
  } else if (lhs_rank == 3 && rhs_rank == 3) {
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, outShape, rewriter.getF32Type());
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(rewriter.getF32Type(), 0.0));
    Value zeroFill =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
    Value matmul =
        rewriter
            .create<linalg::BatchMatmulOp>(loc, zeroFill.getType(),
                                           ValueRange{lhs, rhs}, zeroFill)
            .getResult(0);
    rewriter.replaceOp(op, matmul);
  } else {
  }
}

} // namespace tpu_mlir
