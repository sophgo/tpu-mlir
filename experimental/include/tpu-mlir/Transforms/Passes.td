//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//

#ifndef TPUMLIR_CODEGEN_PASSES
#define TPUMLIR_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"


def DecomposeLinalgGeneric : Pass<"codegen-decompose-linalg-generic", ""> {
  let summary = "Decomposes linalg generic ops into individual ops";
  let constructor = "tpu_mlir::createDecomposeLinalgGenericPass()";
}

def TileAndFuseGreedily : Pass<"codegen-tile-and-fuse-greedily", "func::FuncOp"> {
  let summary = "Tile Consumer And Fuse Producer Greedily with multiCore aware.";
  let constructor = "tpu_mlir::createTileAndFuseGreedilyPass()";
  let options = [
    Option<"coreNum", "core-num",
           "int", /*default=*/"1",
           "The number of cores used to parallel compute.">
  ];
}

def Bufferize : Pass<"codegen-bufferize", "ModuleOp"> {
  let summary = "Convert Tensor to Memref.";
  let constructor = "tpu_mlir::createBufferizePass()";
}

def InstrctionSelction  : Pass<"ISel", "ModuleOp"> {
  let summary = "lowering to Atomic dialect.";
  let constructor = "tpu_mlir::createInstrctionSelctionPass()";
}


#endif // TPUMLIR_CODEGEN_PASSES
