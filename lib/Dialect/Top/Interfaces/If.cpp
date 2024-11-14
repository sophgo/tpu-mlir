//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::IfOp::getFLOPs() { return 0; }

LogicalResult top::IfOp::init(InferenceParameter &p) { return success(); }
void top::IfOp::deinit(InferenceParameter &p) {}

LogicalResult top::IfOp::inference(InferenceParameter &p) {
  if (p.inputs[0][0] > 0)
    return success(); // then_branch
  else
    return failure(); // else_branch
}

void top::IfOp::shape_inference() {
  auto yield_op = getRegion(0).back().getTerminator();
  std::vector<std::vector<int64_t>> shapes;
  // get shape
  for (auto opd : yield_op->getOperands()) {
    shapes.push_back(module::getShape(opd).vec());
  }
  // check if is vaild
  for (uint32_t i = 1; i < getNumRegions(); i++) {
    yield_op = getRegion(i).back().getTerminator();
    auto nof_inputs = yield_op->getNumOperands();
    ASSERT_THIS(nof_inputs == shapes.size() &&
                "Regions have different num of output, fix me.");
    for (uint32_t j = 0; j < nof_inputs; j++) {
      auto _shape = module::getShape(yield_op->getOperand(j)).vec();
      ASSERT_THIS((shapes[j] == _shape) &&
                  "Regions have different output shape, fix me.");
    }
  }
  // set shape
  for (auto res_shape : llvm::zip(getResults(), shapes)) {
    auto res = std::get<0>(res_shape);
    auto shape = std::get<1>(res_shape);
    module::setShapeOrVerify(res, shape);
  }
  return;
}

static inline bool areCompatibleIfTypes(Type ifResultType,
                                        Type branchResultType) {
  if (ShapedType ifShapedType = ifResultType.dyn_cast<ShapedType>()) {
    if (ShapedType branchShapedType = branchResultType.dyn_cast<ShapedType>()) {
      // if quantizedType, just check branch result type if it is also
      // quantizedType
      if (ifShapedType.getElementType().isa<mlir::quant::QuantizedType>())
        return branchShapedType.getElementType()
            .isa<mlir::quant::QuantizedType>();
      else
        return ifShapedType.getElementType() ==
               branchShapedType.getElementType();
    } else {
      return false;
    }
  }

  llvm_unreachable("areCompatibleIfTypes called with non tensor type");
}

LogicalResult top::IfOp::verify() {
  size_t ifNumResults = getNumResults();
  ASSERT_THIS(ifNumResults == getOutput().size() && "output() != all results");
  auto thenResults = getThenBranch().back().getTerminator()->getOperands();
  if (ifNumResults != thenResults.size())
    return emitOpError() << "then branch #results=" << thenResults.size()
                         << " differ from if #results=" << ifNumResults;
  auto elseResults = getElseBranch().back().getTerminator()->getOperands();
  if (ifNumResults != elseResults.size())
    return emitOpError() << "else branch #results=" << elseResults.size()
                         << " differ from if #results=" << ifNumResults;
  auto thenResultTypes = thenResults.getTypes();
  auto elseResultTypes = elseResults.getTypes();
  for (size_t i = 0; i < ifNumResults; ++i) {
    Type ifResultType = getResultTypes()[i];
    if (!areCompatibleIfTypes(ifResultType, thenResultTypes[i]))
      emitOpError() << "then branch disagrees on result type #" << (i + 1)
                    << " of " << ifNumResults;
    if (!areCompatibleIfTypes(ifResultType, elseResultTypes[i]))
      emitOpError() << "else branch disagrees on result type #" << (i + 1)
                    << " of " << ifNumResults;
  }
  return success();
}
