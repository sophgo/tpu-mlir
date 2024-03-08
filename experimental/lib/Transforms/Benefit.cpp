//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu-mlir/Transforms/Benefit.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tpu-mlir/Dialect/BM1690/IR/StructuredOpsInterfaces.h"
#include "tpu-mlir/Transforms/StructuredTransform.h"
#include "llvm/ADT/TypeSwitch.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "iselBenefit"

namespace tpu_mlir {

using namespace mlir;

AffineMap mapShape(linalg::LinalgOp op) {
  SmallVector<int64_t> dimsMap(op.getIndexingMapsArray()[0].getNumDims(), 0);
  for (auto const &[opd, indexMap] :
       llvm::zip(op->getOpOperands(), op.getIndexingMapsArray())) {
    for (auto [expr, dim] :
         llvm::zip(indexMap.getResults(), op.getShape(&opd))) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        auto position = dimExpr.getPosition();
        dimsMap[position] = std::max(dimsMap[position], dim);
      }
    }
  }
  auto context = op.getContext();
  auto results = llvm::to_vector(llvm::map_range(
      dimsMap, [&](int64_t dim) { return getAffineDimExpr(dim, context); }));
  return AffineMap::get(dimsMap.size(), 0, results, context);
}

SmallVector<int64_t> TransformBenefit::getCycle(Transforms &transforms,
                                                SmallVector<int64_t> shape) {
  SmallVector<int64_t> cycles;
  if (transforms.isRoot())
    for (auto &ts : transforms.getChildren())
      getCycle(ts, shape, cycles, 1);
  else
    getCycle(transforms, shape, cycles, 1);

  return cycles;
}

void TransformBenefit::getCycle(Transforms &transforms,
                                SmallVector<int64_t> shape,
                                SmallVector<int64_t> &cycles, int64_t cycle) {
  Transform *transform = &transforms.getTransfom();

  SmallVector<int64_t> new_shape =
      llvm::TypeSwitch<Transform *, SmallVector<int64_t>>(transform)
          .Case([&](Unroll *unroll) {
            auto pos = unroll->dimention.getPosition();
            cycle *= shape[pos];
            shape.erase(shape.begin() + pos);
            return shape;
          })
          .Case([&](MergeDims *mergeDims) {
            auto pos1 = mergeDims->dim1.getPosition();
            auto pos2 = mergeDims->dim2.getPosition();
            shape[pos1] *= shape[pos2];
            shape.erase(shape.begin() + pos2);
            return shape;
          })
          .Case([&](ExpandDims *expandDims) {
            if (expandDims->iteratorType == utils::IteratorType::parallel)
              shape.insert(shape.begin(), 1);
            else
              shape.push_back(1);
            return shape;
          })
          .Default(std::move(shape));

  LLVM_DEBUG(transform->dump(); interleaveComma(new_shape, llvm::dbgs());
             llvm::dbgs() << "\n";);

  if (transforms.empty()) {
    auto acceleration = target.getAccelerateMap().getResults();
    for (auto [dim, accExpr] : llvm::zip(new_shape, acceleration)) {
      auto acc = cast<AffineConstantExpr>(accExpr).getValue();
      cycle *= (dim + acc - 1) / acc;
    }
    LLVM_DEBUG(llvm::dbgs() << "End: " << cycle << "\n\n";);
    return cycles.push_back(cycle);
  }

  for (auto &ts : transforms) {
    getCycle(ts, new_shape, cycles, cycle);
  }
}

} // namespace tpu_mlir
