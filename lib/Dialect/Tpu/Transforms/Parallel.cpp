//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {
extern void populateParalleBM1684XPatterns(RewritePatternSet *patterns,
                                           int coreNum);

// Generate index vector for a given shape, it treats this shape as a nested
// loop.
// eg: given a shape (2, 3), it will call the function with (0, 0) (0, 1) ...
// (1, 2)
template <typename T, class Func>
void invokeInIterationSpace(ArrayRef<T> shape, Func &&func,
                            SmallVector<T, 8> dims = {}) {
  auto dim = dims.size();
  SmallVector dimN(dims);
  dimN.push_back(0);
  for (T i = 0, n = shape[dim]; i < n; i++) {
    dimN.back() = i;
    if (dimN.size() < shape.size()) {
      invokeInIterationSpace(shape, func, dimN);
    } else {
      func(dimN);
    }
  }
}

// convert <(d0, d1) -> (d0)> to [1, 0] when the shape is 2 dimension.
// convert <(d0, d1) -> (d0)> to [1] when the shape is 1 dimension.
template <typename T>
auto getValidStride(Attribute indexMapAttr, ArrayRef<T> iterShape) {
  // filter valid shape
  auto indexMap = cast<AffineMapAttr>(indexMapAttr).getValue();
  SmallVector<T> strides(iterShape.size(), 0);
  if (indexMap.getNumResults() == 0)
    return strides;
  auto context = indexMap.getContext();
  for (int i = iterShape.size() - 1, stride = 1; i >= 0; i--) {
    if (auto outIndex =
            indexMap.getResultPosition(getAffineDimExpr(i, context))) {
      strides[i] = stride;
      stride *= iterShape[i];
    }
  }
  return strides;
}

template <typename T>
inline T getValidIndex(ArrayRef<T> dims, ArrayRef<T> strides) {
  T index = 0;
  for (auto [a, b] : llvm::zip(dims, strides))
    index += a * b;
  return index;
}

// get the types of split value.
// For each operands and results of the compute operation, we need to split them
// based on indexing map. If the operand use many times, here just create one.
// eg: res<2x64x...> = Conv(opd0<2x...>, opd1<64x...>)
// will split to 4 expressions: res<1x32x...> = Conv(opd0<1x...>, opd1<32x...>)
// but we only need to create 2 opd0<1x...> and opd1<32x...>
std::optional<SmallVector<Type>> getSplitTypes(Attribute valueMap, Value value,
                                               ArrayRef<int64_t> shapeParallel,
                                               int splitDim, int splitMax) {
  auto vMap = cast<AffineMapAttr>(valueMap).getValue();
  auto context = valueMap.getContext();
  int index;
  if (auto indexOpt =
          vMap.getResultPosition(getAffineDimExpr(splitDim, context))) {
    index = indexOpt.value();
  } else {
    if (vMap.getNumResults() == 0) // NoneOp or use all the data of the value.
      return std::nullopt;
    // We do not slice this value but use index iteration.
    bool find = false;
    for (int i = splitDim - 1; i >= 0; i--) {
      if (auto indexOpt =
              vMap.getResultPosition(getAffineDimExpr(i, context))) {
        index = indexOpt.value();
        splitDim = i;
        splitMax = 1;
        find = true;
        break;
      }
    }
    if (!find)
      return std::nullopt;
  }
  auto shape = SmallVector<int64_t>(module::getShape(value));
  int parallelSpace = 1;
  for (int i = 0; i < index; i++) {
    parallelSpace *= shape[i];
    shape[i] = 1;
  }
  SmallVector<Type> outputType;
  auto dtype = module::getElementType(value);
  for (int i = 0; i < parallelSpace; i++) {
    for (int j = 0, n = shapeParallel[splitDim]; j < n; j += splitMax) {
      if (j + splitMax <= n)
        shape[index] = splitMax;
      else
        shape[index] = n - j;
      outputType.push_back(RankedTensorType::get(shape, dtype));
    }
  }
  return outputType;
};

bool forAll(IndexingMapsInterface op, int num_core = 1) {
  auto indexMap = op.getIndexingMaps();
  if (indexMap.size() == 0)
    return false;

  // for each dim slice, we should create many operations for each of them.
  // treat [dim0, dim1, ..] as an integer dim0*dim1 and we can only decompose
  // the low dimension to a*b^ which a*b^ should be equal to dim1，now we have
  // dim0*a iteration space to travel with each have b^ elements.

  // split operandsMap and resultsMap
  auto operandsMap = indexMap.getValue().slice(0, op->getNumOperands());
  auto resultsMap =
      indexMap.getValue().slice(op->getNumOperands(), op->getNumResults());

  // use the first resultsMap as a sample, each AffineMap in indexingMap have
  // the same dimCount.
  auto resultMap = cast<AffineMapAttr>(resultsMap[0]).getValue();
  if (!resultMap.isIdentity())
    return false;

  // :load balance:
  // shape = [a, b]; other situations can be reduced to this formula.
  // a * b = a * \sum_{i=1}^n (b_i)
  // a * n <= num_core
  // Find the largest n
  // This implement use the maxSlice as much as possible, but it does not take
  // the number of NPU into account. #please improve This
  auto shapeParallel =
      module::getShape(op->getResult(0)).slice(0, resultMap.getNumInputs());

  int splitDim = 0, splitMax = 1;
  SmallVector<int64_t, 4> iterationShape;
  for (int64_t i = 0, n = shapeParallel.size(), iterSpace = 1; i < n; ++i) {
    if (iterSpace * shapeParallel[i] >= num_core) {
      splitDim = i;
      int coreK = num_core / iterSpace;                  // This is the lower n
      splitMax = (shapeParallel[i] + coreK - 1) / coreK; // This is max(b_i)
      auto n = (shapeParallel[i] + splitMax - 1) / splitMax;
      iterationShape.push_back(n);
      break;
    } else {
      iterationShape.push_back(shapeParallel[i]);
      iterSpace *= shapeParallel[i];
    }
  }

  if (splitDim == 0 && iterationShape[0] == 1)
    return false;

  auto rewriter = IRRewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  auto parallelOp = rewriter.create<tpu::ParallelOp>(
      op.getLoc(), op->getResultTypes(), op->getOperands());
  auto body = new Block();
  parallelOp.getBody().push_back(body);
  rewriter.setInsertionPointToStart(body);
  rewriter.replaceAllUsesWith(op->getResults(), parallelOp.getResults());

  // Travel the multi-dimensional iteration space.
  // 1. build split operation for each operand.
  SmallVector<Operation *, 4> splitOps;
  SmallVector<SmallVector<int64_t, 4>, 4> operandsStride;
  for (auto [index, valueMap, value] :
       llvm::enumerate(operandsMap, op->getOperands())) {
    if (auto outTypes = getSplitTypes(valueMap, value, ArrayRef(shapeParallel),
                                      splitDim, splitMax)) {
      auto name = module::getName(value) + "_" + Twine(index);
      auto nameLoc = NameLoc::get(rewriter.getStringAttr(name));

      splitOps.push_back(rewriter.create<tpu::SplitOp>(
          nameLoc, TypeRange(outTypes.value()), value));
    } else {
      splitOps.push_back(value.getDefiningOp());
    }
    operandsStride.push_back(
        getValidStride(valueMap, ArrayRef(iterationShape)));
  }

  // 2. build distributing compute operation for each num_core.
  SmallVector<Operation *> computeOps;
  SmallVector<SmallVector<Type>> outputsTypes;
  for (auto [valueMap, value] : llvm::zip(resultsMap, op->getResults())) {
    outputsTypes.push_back(getSplitTypes(valueMap, value,
                                         ArrayRef(shapeParallel), splitDim,
                                         splitMax)
                               .value());
  }

  auto resultStride = getValidStride(resultsMap[0], ArrayRef(iterationShape));

  auto createComputeOp = [&](ArrayRef<int64_t> dims) {
    SmallVector<Value, 4> operands;
    for (auto [index, spOp] : llvm::enumerate(splitOps)) {
      ArrayRef stride(operandsStride[index]);
      operands.push_back(spOp->getResult(getValidIndex(dims, stride)));
    }

    SmallVector<Type, 2> resultsType;
    for (auto types : outputsTypes)
      resultsType.push_back(types[getValidIndex(dims, ArrayRef(resultStride))]);

    auto suffix =
        llvm::formatv("{0:$[_]}", make_range(dims.begin(), dims.end()));
    auto name = module::getName(op, 0) + "_" + suffix;
    auto nameLoc = NameLoc::get(rewriter.getStringAttr(name));

    computeOps.push_back(rewriter.create(nameLoc, op->getName().getIdentifier(),
                                         operands, resultsType,
                                         op->getAttrs()));
  };
  // unroll iteration space
  invokeInIterationSpace(ArrayRef(iterationShape), createComputeOp);

  // 3. join the computation from multi-num_core.
  SmallVector<Value> joinValues;
  for (int i = 0, n = outputsTypes.size(); i < n; i++) {
    SmallVector<Value> operands;
    for (auto cpOp : computeOps) {
      operands.push_back(cpOp->getResult(i));
    }
    joinValues.push_back(rewriter
                             .create<tpu::JoinOp>(op->getLoc(),
                                                  op->getResultTypes()[i],
                                                  operands)
                             .getResult());
  }
  rewriter.create<tpu::YieldOp>(op->getLoc(), joinValues);
  // cleanup
  rewriter.eraseOp(op);
  return true;
};

class ParallelPass : public ParallelBase<ParallelPass> {
public:
  ParallelPass() {}
  void runOnOperation() override {
    module::setCoreNum(num_core);
    if (num_core < 2) {
      return;
    }
    auto modules = module::getAllModules();
    for (auto m : *modules) {
      // run each submodule
      RewritePatternSet patterns(&getContext());
      if (module::isBM1684XFamily() || module::isSG2260Family()) {
        populateParalleBM1684XPatterns(&patterns, num_core);
      }
      auto config = GreedyRewriteConfig();
      config.maxIterations = 1; // apply each pattern only once.
      applyPatternsAndFoldGreedily(m, std::move(patterns), config);
      m->walk([&](IndexingMapsInterface op) {
        if (module::isOpInGroup(op) || module::isOpInParallel(op))
          return;
        forAll(op, num_core);
      });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createParallelPass() {
  return std::make_unique<ParallelPass>();
}
} // namespace tpu
} // namespace tpu_mlir
