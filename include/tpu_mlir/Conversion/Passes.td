//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_CONVERSION_PASSES
#define TPU_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TopToTpu
//===----------------------------------------------------------------------===//

def ConvertTopToTpu : Pass<"convert-top-to-tpu", "ModuleOp"> {
  let summary = "Convert top-level Top Ops to Tpu Ops";
  let constructor = "tpu_mlir::createConvertTopToTpu()";
  let dependentDialects = ["tpu_mlir::top::TopDialect", "tpu_mlir::tpu::TpuDialect"];
  let options = [
    Option<"mode", "mode", "std::string", /*default=*/"",
           "default quantization mode: INT8/BF16/F32">,
    Option<"qtable", "qtable", "std::string", /*default=*/"",
           "a table of Ops that quantized to specific mode">,
    Option<"isAsymmetric", "asymmetric", "bool", /*default=*/"false",
           "true for asymmetric quantization, or false for symmetric">,
    Option<"doWinograd", "doWinograd", "bool", /*default=*/"false",
           "true for trying winograd ,or false for not">,
    Option<"ignore_f16_overflow", "ignore_f16_overflow", "bool", /*default=*/"false",
           "some ops convert from f16 to f32, to avoid f16 overflow">,
    Option<"weightFileName", "weightFileName", "std::string", /*default=*/"",
           "weight file name to save">,
    Option<"LinearQuantMode", "linear_quant_mode", "std::string", /*default=*/"",
           "W4A16/ W8A16 quant mode for linear layer">
  ];
}


//===----------------------------------------------------------------------===//
// TopToTosa
//===----------------------------------------------------------------------===//
def ConvertTopToTosa : Pass<"convert-top-to-tosa", "ModuleOp"> {
  let summary = "Convert top-level Top Ops to Tosa Ops";
  let constructor = "tpu_mlir::createConvertTopToTosa()";
  let dependentDialects = ["tpu_mlir::top::TopDialect", "mlir::tosa::TosaDialect"];
  let options = [
    Option<"includeWeight", "includeWeight", "bool", /*default=*/"false",
           "true for including weight datas in tosa.mlir, or false for not">,
  ];
}

#endif // TPU_MLIR_CONVERSION_PASSES
