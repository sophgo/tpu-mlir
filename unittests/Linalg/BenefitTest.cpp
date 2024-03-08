//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu-mlir/Transforms/Benefit.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "tpu-mlir/Transforms/StructuredTransform.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace tpu_mlir;

void printComputeType(const ComputePattern &CType) {

  llvm::errs() << "\n[\n";
  for (auto i : CType.iteratorTypes) {
    if (i == utils::IteratorType::parallel)
      llvm::errs() << "  P";
    else
      llvm::errs() << "  R";
  }
  llvm::errs() << "\n";
  for (auto i : CType.indexingMaps) {
    llvm::errs() << "  ";
    i.dump();
  }
  llvm::errs() << "]\n";
}

ArrayAttr getIndexingMaps(std::string ASM, MLIRContext *context) {
  return cast<ArrayAttr>(mlir::parseAttribute(ASM, context));
}

SmallVector<AffineMap> getIndexingMapsVector(std::string ASM,
                                             MLIRContext *context) {
  return llvm::to_vector(getIndexingMaps(ASM, context)
                             .getAsValueRange<AffineMapAttr, AffineMap>());
}

TEST(Benefit, Cycle) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm1 = R"mlir([
            affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>,
            affine_map<(d0, d1, d2, d3) -> (0, d1, d2, 0)>,
            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                              ])mlir";

  std::string affineMapAsm2 = R"mlir([
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                              ])mlir";

  auto affineMaps1 = getIndexingMapsVector(affineMapAsm1, &context);
  auto affineMaps2 = getIndexingMapsVector(affineMapAsm2, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  // auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes1{P, P, P, P};
  SmallVector<utils::IteratorType> iteratorTypes2{P, P, P, P};
  auto source = tpu_mlir::ComputePattern{affineMaps1, iteratorTypes1};
  auto target = tpu_mlir::ComputePattern{affineMaps2, iteratorTypes2};

  auto s = Solver(target, 4);
  auto out = s.solve(source);
  // out.dump();
  EXPECT_TRUE(out.size() > 0);
  auto mm = bm1690::registerTraits(&context);

  auto a = TransformBenefit(mm["bm1690.arithmetic.and"]);
  a.getCycle(out, {128, 32, 15, 30});
  // llvm::errs() << out.size() << "/" << s.getAllPath() << "\n";
}
