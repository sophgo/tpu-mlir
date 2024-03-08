//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "tpu-mlir/Dialect/BM1690/IR/BM1690.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace mlir;

using namespace mlir::detail;

namespace {
TEST(BM1690IR, MatMulOp) {
  using namespace tpu_mlir;

  DialectRegistry registry;
  registry.insert<bm1690::BM1690Dialect, func::FuncDialect>();

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  OpBuilder builder(&context);
  auto uloc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(uloc);
  builder.setInsertionPointToEnd(theModule.getBody());
  auto inputType = MemRefType::get({2, 512, 1, 64}, builder.getF16Type());
  llvm::SmallVector<mlir::Type, 4> argTypes(2, inputType);

  auto id1 = builder.getType<bm1690::DMAIdType>(1);
  argTypes.push_back(id1);
  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
  auto function = builder.create<func::FuncOp>(uloc, "main", funcType);
  builder.setInsertionPointToEnd(function.addEntryBlock());

  auto id2 = builder.getType<bm1690::TIUIdType>(2);
  SmallVector<Type> outTypes{inputType, id2};

  auto mat0 = builder.create<bm1690::MatMulOp>(
      uloc, outTypes, function.getArgument(0), function.getArgument(1), nullptr,
      function.getArgument(2), true, false, true);

  auto id3 = builder.getType<bm1690::TIUIdType>(3);

  builder.create<bm1690::MatMulOp>(
      uloc, inputType, id3, function.getArgument(0), mat0.getResult(), nullptr,
      function.getArgument(2), true, false, true);
  builder.create<func::ReturnOp>(uloc);

  auto &reg = mat0.getProperties().reg;
  EXPECT_EQ(reg.opd0_n, 0);
  EXPECT_EQ(reg.opd0_c, 0);
  EXPECT_EQ(reg.opd0_w, 0);
  EXPECT_EQ(reg.cmd_id_dep, 0);

  mat0.verifyAndCodeGen(); // set the register

  EXPECT_EQ(reg.opd0_n, 2);
  EXPECT_EQ(reg.opd0_c, 512);
  EXPECT_EQ(reg.opd0_w, 64);
  EXPECT_EQ(reg.cmd_id_dep, 1);
  // theModule.dump();
}

TEST(BM1690IR, ConvOp) {
  using namespace tpu_mlir;

  DialectRegistry registry;
  registry.insert<bm1690::BM1690Dialect, func::FuncDialect>();

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  OpBuilder builder(&context);
  auto uloc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(uloc);
  builder.setInsertionPointToEnd(theModule.getBody());
  auto inputType = MemRefType::get({2, 8, 64, 64}, builder.getF16Type());
  llvm::SmallVector<mlir::Type, 4> argTypes(2, inputType);

  auto id1 = builder.getType<bm1690::DMAIdType>(1);
  argTypes.push_back(id1);
  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
  auto function = builder.create<func::FuncOp>(uloc, "main", funcType);
  builder.setInsertionPointToEnd(function.addEntryBlock());

  auto id2 = builder.getType<bm1690::TIUIdType>(2);
  SmallVector<int64_t> kernel{3, 3};
  SmallVector<int64_t> dialation{3, 3};
  SmallVector<int64_t> stride{3, 3};
  SmallVector<int64_t> insertion{3, 3};
  SmallVector<int64_t> pads{0, 0, 0, 0};
  auto conv0 = builder.create<bm1690::ConvOp>(
      uloc, inputType, id2, function.getArgument(0), function.getArgument(1),
      nullptr, function.getArgument(2), kernel, stride, dialation, insertion,
      pads, false, false, bm1690::PaddingMode::constant);

  auto id3 = builder.getType<bm1690::TIUIdType>(3);
  builder.create<bm1690::ConvOp>(
      uloc, inputType, id3, conv0.getResult(), function.getArgument(1), nullptr,
      function.getArgument(2), kernel, stride, dialation, insertion,
      pads, false, false, bm1690::PaddingMode::constant);
  builder.create<func::ReturnOp>(uloc);

  auto &reg = conv0.getProperties().reg;
  EXPECT_EQ(reg.res0_n, 0);
  EXPECT_EQ(reg.opd0_c, 0);
  EXPECT_EQ(reg.opd0_w, 0);
  EXPECT_EQ(reg.cmd_id_dep, 0);

  conv0.verify(); // set the register

  EXPECT_EQ(reg.res0_n, 2);
  EXPECT_EQ(reg.opd0_c, 8);
  EXPECT_EQ(reg.opd0_h, 64);
  EXPECT_EQ(reg.opd0_w, 64);
  EXPECT_EQ(reg.cmd_id_dep, 1);
  // theModule.dump();
}

TEST(BM1690IR, DMATensorOp) {
  using namespace tpu_mlir;

  DialectRegistry registry;
  registry.insert<bm1690::BM1690Dialect, func::FuncDialect>();

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  OpBuilder builder(&context);
  auto uloc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(uloc);
  builder.setInsertionPointToEnd(theModule.getBody());
  auto inputType = MemRefType::get({2, 8, 64, 64}, builder.getF16Type());
  llvm::SmallVector<mlir::Type, 4> argTypes(2, inputType);

  auto id1 = builder.getType<bm1690::TIUIdType>(1);
  argTypes.push_back(id1);
  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
  auto function = builder.create<func::FuncOp>(uloc, "main", funcType);
  builder.setInsertionPointToEnd(function.addEntryBlock());

  auto id2 = builder.getType<bm1690::DMAIdType>(2);

  auto dma0 = builder.create<bm1690::DMATensorOp>(
      uloc, inputType, id2, function.getArgument(0), function.getArgument(2));

  auto id3 = builder.getType<bm1690::TIUIdType>(3);

  auto mat0 = builder.create<bm1690::MatMulOp>(
      uloc, inputType, id3, dma0.getTarget(), function.getArgument(1), nullptr,
      dma0.getId(), true, false, true);

  auto id4 = builder.getType<bm1690::DMAIdType>(4);
  auto dma1 = builder.create<bm1690::DMATensorTransOp>(
      uloc, inputType, id4, mat0.getResult(), mat0.getId());

  auto id5 = builder.getType<bm1690::DMAIdType>(5);

  builder.create<bm1690::DMATensorBroadcastOp>(uloc, inputType, id5,
                                               dma1.getTarget(), mat0.getId());

  builder.create<func::ReturnOp>(uloc);

  auto &reg = dma0.getProperties().reg;
  EXPECT_EQ(reg.dst_nsize, 0);
  EXPECT_EQ(reg.dst_csize, 0);
  EXPECT_EQ(reg.dst_hsize, 0);
  EXPECT_EQ(reg.cmd_id_dep, 0);

  dma0.verify(); // set the register

  EXPECT_EQ(reg.dst_nsize, 2);
  EXPECT_EQ(reg.src_csize, 8);
  EXPECT_EQ(reg.cmd_id_dep, 1);
  // theModule.dump();
}

}
