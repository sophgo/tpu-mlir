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
#include "tpu_mlir/Dialect/SG2260/IR/SG2260.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"

using namespace mlir;

using namespace mlir::detail;

namespace {
TEST(SG2260IR, MatMulOp) {
  using namespace tpu_mlir;

  DialectRegistry registry;
  registry.insert<sg2260::SG2260Dialect, func::FuncDialect>();

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  OpBuilder builder(&context);
  auto uloc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(uloc);
  builder.setInsertionPointToEnd(theModule.getBody());
  auto inputType = MemRefType::get({2, 512, 1, 64}, builder.getF16Type());
  llvm::SmallVector<mlir::Type, 4> argTypes(2, inputType);

  auto id1 = builder.getType<sg2260::DMAIdType>(1);
  argTypes.push_back(id1);
  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
  auto function = builder.create<func::FuncOp>(uloc, "main", funcType);
  builder.setInsertionPointToEnd(function.addEntryBlock());

  auto id2 = builder.getType<sg2260::TIUIdType>(2);
  SmallVector<Type> outTypes{inputType, id2};

  auto mat0 = builder.create<sg2260::MatMulOp>(
      uloc, outTypes, function.getArgument(0), function.getArgument(1), nullptr,
      function.getArgument(2), true, false, true);

  auto id3 = builder.getType<sg2260::TIUIdType>(3);

  builder.create<sg2260::MatMulOp>(
      uloc, inputType, id3, function.getArgument(0), mat0.getResult(), nullptr,
      function.getArgument(2), true, false, true);
  builder.create<func::ReturnOp>(uloc);

  auto &reg = mat0.getProperties().reg;
  EXPECT_EQ(reg.opd0_n, 0);
  EXPECT_EQ(reg.opd0_c, 0);
  EXPECT_EQ(reg.opd0_w, 0);
  EXPECT_EQ(reg.cmd_id_dep, 0);

  mat0.verify(); // set the register

  EXPECT_EQ(reg.opd0_n, 2);
  EXPECT_EQ(reg.opd0_c, 512);
  EXPECT_EQ(reg.opd0_w, 64);
  EXPECT_EQ(reg.cmd_id_dep, 1);
}

} // namespace
