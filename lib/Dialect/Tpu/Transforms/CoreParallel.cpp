//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/CoreParallel/CoreParallel.hpp"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

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

// forAll will split the computation to multiple cores which in the range of
// [offset, offset+num_core)
tpu::CoreParallelOp forAll(IndexingMapsInterface op, int offset = 0,
                           int num_core = 1) {
  if (getRunMode(op) == RunMode::TPU_DYNAMIC) {
    return nullptr;
  }
  auto indexMap = op.getIndexingMaps();
  if (!indexMap || indexMap.empty())
    return nullptr;

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
  if (resultMap.isEmpty())
    return nullptr;

  // :load balance:
  // shape = [a, b]; other situations can be reduced to this formula.
  // a * b = a * \sum_{i=1}^n (b_i)
  // a * n <= num_core
  // Find the largest n
  // This implement use the maxSlice as much as possible, but it does not take
  // the number of NPU into account. #please improve This
  auto shapeParallel = SmallVector<int64_t>(
      module::getShape(op->getResult(0)).slice(0, resultMap.getNumInputs()));

  int splitDim = 0, splitMax = 1;
  SmallVector<int64_t, 4> iterationShape;

  { // This is a temporary fix for GroupNorm support; Try to refactor this out.
    if (auto groupNormOp = dyn_cast<tpu::GroupNormOp>(op.getOperation())) {
      shapeParallel[1] = groupNormOp.getNumGroups();
    }
  }

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
    return nullptr;

  { // This is a temporary fix for GroupNorm support; Try to refactor this out.
    if (auto groupNormOp = dyn_cast<tpu::GroupNormOp>(op.getOperation())) {
      auto channel = module::getShape(groupNormOp.getInput())[1];
      shapeParallel[1] = channel;
      if (splitDim == 1)
        splitMax *= channel / groupNormOp.getNumGroups();
    }
  }

  auto rewriter = IRRewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  auto parallelOp = rewriter.create<tpu::CoreParallelOp>(
      op.getLoc(), op->getResultTypes(), op->getOperands(), offset, num_core);
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
      auto name = module::getName(value) + "_" + Twine(index).str();
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
      if (spOp)
        operands.push_back(spOp->getResult(getValidIndex(dims, stride)));
      else // inputs
        operands.push_back(op->getOperand(index));
    }

    SmallVector<Type, 2> resultsType;
    for (auto types : outputsTypes)
      resultsType.push_back(types[getValidIndex(dims, ArrayRef(resultStride))]);

    auto suffix =
        llvm::formatv("_{0:$[_]}", make_range(dims.begin(), dims.end()));
    auto name = module::getName(op, 0) + suffix.str().c_str();
    auto nameLoc = NameLoc::get(rewriter.getStringAttr(name));

    computeOps.push_back(rewriter.create(nameLoc, op->getName().getIdentifier(),
                                         operands, resultsType,
                                         op->getAttrs()));
    { // This is a temporary fix for GroupNorm support; Try to refactor this
      // out.
      if (auto groupNormOp = dyn_cast<tpu::GroupNormOp>(computeOps.back())) {
        auto numGroup = groupNormOp.getNumGroups();
        auto itemPerGroup = module::getShape(op->getOperand(0))[1] / numGroup;
        auto channel = module::getShape(groupNormOp.getInput())[1];
        groupNormOp.setNumGroups(channel / itemPerGroup);
      }
    }
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
  return parallelOp;
};

bool packOperation(IndexingMapsInterface op, int offset = 0, int num_core = 1) {

  auto rewriter = IRRewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  auto parallelOp = rewriter.create<tpu::CoreParallelOp>(
      op.getLoc(), op->getResultTypes(), op->getOperands(), offset, num_core);
  auto body = new Block();
  parallelOp.getBody().push_back(body);
  rewriter.setInsertionPointToStart(body);
  rewriter.replaceAllUsesWith(op->getResults(), parallelOp.getResults());

  auto yieldOp = rewriter.create<tpu::YieldOp>(op->getLoc(), op->getResults());
  op->moveBefore(yieldOp);
  return true;
};

void groupParallelDistribute(tpu::GroupParallelOp op, int num_core) {
  int regionNum = op.getNumRegions();

  assert(
      regionNum <= num_core &&
      "The count of parallel subgraphs must not exceed the number of cores.");

  // all the job can be distributed to different core.
  if (regionNum == num_core)
    return;
  int coresInOneRegion = num_core / regionNum;
  int remaining = num_core % regionNum;
  int start = 0;

  SmallVector<std::array<int, 2>> coresInRegion;
  for (auto [index, region] : llvm::enumerate(op.getParallel())) {
    int jobs = coresInOneRegion + (index < remaining);
    for (auto coreParallelOp : region.getOps<IndexingMapsInterface>()) {
      forAll(coreParallelOp, start, jobs);
    }
    coresInRegion.push_back({start, jobs});
    start += jobs;
  }

  SmallVector<std::array<Region::OpIterator, 2>> oPit;
  for (auto [index, region] : llvm::enumerate(op.getParallel())) {
    oPit.push_back({region.getOps().begin(), region.getOps().end()});
  }

  // Ensure that coreParallelOps are uniformly symmetrical across all regions to
  // simplify the code generation process.
  while (oPit[0][0] != oPit[0][1]) {
    if (any_of(oPit,
               [](auto &it) { return isa<tpu::CoreParallelOp>(*it[0]); })) {
      for (auto [index, coreRange] : llvm::enumerate(coresInRegion)) {
        auto &it = oPit[index];
        if (!isa<tpu::CoreParallelOp>(*it[0]) &&
            true == isa<IndexingMapsInterface>(*it[0])) {
          packOperation(cast<IndexingMapsInterface>(*it[0]), coreRange[0],
                        coreRange[1]);
        }
        if (index > 0)
          it[0]++;
      }
    }
    oPit[0][0]++;
  }
}

class CoreParallelPass : public CoreParallelBase<CoreParallelPass> {
public:
  CoreParallelPass() = default;
  void runOnOperation() override {
    auto num_core = module::getCoreNum();
    if (num_core < 2) {
      return;
    }
    auto modules = module::getAllModules();
    for (auto m : *modules) {
      // do core match first
      doCoreParallelPattern(m);
      // do specific
      doSpecificPattern(m);
      // normal situations to multi cores
      m->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<tpu::GroupOp>(op)) {
          return WalkResult::skip();
        }
        if (auto groupParallelOp = dyn_cast<tpu::GroupParallelOp>(op)) {
          groupParallelDistribute(groupParallelOp, num_core);
          return WalkResult::skip();
        }
        if (supportMultiCore(op)) {
          if (auto matmul_op = dyn_cast<tpu::MatMulOp>(op)) {
            auto l2_buffer_size = matmul_op.getL2BufferSize();
            int64_t l2memSize = backend::BM168x::L2_SRAM_SIZE;
            auto core_num = module::getCoreNum();
            const int MAX_CORES = 8;
            l2memSize = (l2memSize / MAX_CORES) * core_num;
            if (l2_buffer_size <= l2memSize) {
              return WalkResult::skip();
            }
          } else {
            return WalkResult::skip();
          }
        }
        if (auto coreParallelOp = dyn_cast<IndexingMapsInterface>(op)) {
          forAll(coreParallelOp, 0, num_core);
        }
        return WalkResult::advance();
      });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createCoreParallelPass() {
  return std::make_unique<CoreParallelPass>();
}
} // namespace tpu
} // namespace tpu_mlir
