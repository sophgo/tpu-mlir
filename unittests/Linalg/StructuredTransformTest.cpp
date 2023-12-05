//===-- AffineTest.cpp - Affine feature explore ---------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu-mlir/Transforms/StructuredTransform.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"
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

TEST(AffineTranform, Unroll) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1+s0*d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
       )mlir";
  auto affineMaps = getIndexingMapsVector(affineMapAsm, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps, iteratorTypes};

  {
    auto transform = tpu_mlir::Unroll(cast<AffineDimExpr>(d0));
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2)[s0, s1] -> (d0+s0*d1, d2)>,
             affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
                           )mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }

  {
    auto transform = tpu_mlir::Unroll(cast<AffineDimExpr>(d1));
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2)[s0, s1] -> (d0, d1 * s0, d2)>,
             affine_map<(d0, d1, d2) -> (d0, d1, d2)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }

  {
    auto transform = tpu_mlir::Unroll(cast<AffineDimExpr>(d2));
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2)[s0, s1] -> (d0, d1, d2)>,
             affine_map<(d0, d1, d2) -> (d0, d1, d2)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }
}

TEST(AffineTranform, DropSymbol) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1+s0*d2, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
       )mlir";
  auto affineMaps = getIndexingMapsVector(affineMapAsm, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps, iteratorTypes};

  {
    auto transform = tpu_mlir::DropSymbol(cast<AffineSymbolExpr>(s0));
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0] -> (d0, d1 + d2, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }

  {
    auto transform = tpu_mlir::DropSymbol(cast<AffineSymbolExpr>(s1));
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0] -> (d0, d1 + d2 * s0, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }
}

TEST(AffineTranform, MergeDims) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1+s0*d2, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
       )mlir";
  auto affineMaps = getIndexingMapsVector(affineMapAsm, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps, iteratorTypes};

  {
    auto transform =
        tpu_mlir::MergeDims(cast<AffineDimExpr>(d0), cast<AffineDimExpr>(d1));
    auto out = transform.run(source);
    EXPECT_FALSE(out);
  }

  { // fail to merge dim with different iterator type.
    auto transform =
        tpu_mlir::MergeDims(cast<AffineDimExpr>(d2), cast<AffineDimExpr>(d3));
    auto out = transform.run(source);
    EXPECT_FALSE(out);
  }

  {
    auto transform1 = tpu_mlir::Unroll(cast<AffineDimExpr>(d2));
    auto out = transform1.run(source);
    auto transform2 =
        tpu_mlir::MergeDims(cast<AffineDimExpr>(d0), cast<AffineDimExpr>(d1));
    auto out1 = transform2.run(out.value()).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1)[s0, s1] -> (d0, d1)>,
             affine_map<(d0, d1) -> (d0, d1)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out1.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out1.indexingMaps[1]);
  }
}

TEST(AffineTranform, Permutation) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1+s0*d2, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
       )mlir";
  auto affineMaps = getIndexingMapsVector(affineMapAsm, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps, iteratorTypes};

  {
    auto transform = tpu_mlir::Permutation({0, 1}, {true, true});
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d1 + d2 * s0, d0, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }

  // {
  //   auto transform = tpu_mlir::Permutation({0, 1}, {false, true});
  //   auto out = transform.run(source);
  //   printComputeType(out.value());
  // }

  // {
  //   auto transform = tpu_mlir::Permutation({0, 3}, {true, true});
  //   auto out = transform.run(source);
  //   printComputeType(out.value());
  // }
}

TEST(AffineTranform, ExpandDims) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1+s0*d2, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
       )mlir";
  auto affineMaps = getIndexingMapsVector(affineMapAsm, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps, iteratorTypes};

  {
    auto transform =
        tpu_mlir::ExpandDims(utils::IteratorType::parallel, {false, false});
    auto out = transform.run(source);
    EXPECT_FALSE(out);
  }

  {
    auto transform =
        tpu_mlir::ExpandDims(utils::IteratorType::parallel, {true, false});
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2, d3, d4)[s0, s1] -> (d0, d1, d2 + d3 * s0, d3, d4)>,
             affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }

  {
    auto transform =
        tpu_mlir::ExpandDims(utils::IteratorType::reduction, {true, true});
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2, d3, d4)[s0, s1] -> (d0, d1 + d2 * s0, d2, d3, d4)>,
             affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }
}

TEST(AffineTranform, DecomposeExpr) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1+s0*d2, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
       )mlir";
  auto affineMaps = getIndexingMapsVector(affineMapAsm, &context);

  AffineExpr d0, d1, d2, d3, s0, s1;

  bindDims(&context, d0, d1, d2, d3);
  bindSymbols(&context, s0, s1);

  auto P = utils::IteratorType::parallel;
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps, iteratorTypes};

  {
    auto transform = tpu_mlir::DecomposeExpr(0);
    auto out = transform.run(source);
    EXPECT_FALSE(out);
  }

  {
    auto transform = tpu_mlir::DecomposeExpr(1);
    auto out = transform.run(source).value();
    std::string outAsm = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0, d1, d2 * s0, d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                          ])mlir";
    auto outIndexingMaps = getIndexingMapsVector(outAsm, &context);
    EXPECT_EQ(outIndexingMaps[0], out.indexingMaps[0]);
    EXPECT_EQ(outIndexingMaps[1], out.indexingMaps[1]);
  }
}

TEST(AffineTranform, Solver) {

  DialectRegistry registry;
  MLIRContext context(registry);

  Builder builder(&context);

  std::string affineMapAsm1 = R"mlir([
             affine_map<(d0, d1, d2, d3)[s0] -> (d0, d1+s0*d2, d3)>,
             affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>,
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
  auto R = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> iteratorTypes1{P, P, P, R};
  SmallVector<utils::IteratorType> iteratorTypes2{P, P, P, R};
  auto source = tpu_mlir::ComputePattern{affineMaps1, iteratorTypes1};
  auto target = tpu_mlir::ComputePattern{affineMaps2, iteratorTypes2};

  // printComputeType(source);
  // printComputeType(target);

  auto s = Solver(target, 4);
  auto out = s.solve(source);
  out.dump();
  EXPECT_TRUE(out.size() > 0);
  llvm::errs() << out.size() << "/" << s.getAllPath() << "\n";
}
