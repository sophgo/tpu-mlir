// ==============================================================================
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
// ==============================================================================

#ifndef SG2260_ENUMS
#define SG2260_ENUMS


include "mlir/IR/EnumAttr.td"

def SG2260_PaddingMode: I32EnumAttr<"PaddingMode",
    "requant mode supported by PadOp",
    [
      I32EnumAttrCase<"constant", 0>,
      I32EnumAttrCase<"reflect", 1>,
      I32EnumAttrCase<"replication", 2>,
      I32EnumAttrCase<"circular", 3>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::sg2260";
}

def SG2260_RoundMode: I32EnumAttr<"RoundMode",
    "round mode supported by Round",
    [
      I32EnumAttrCase<"HalfToEven", 0>,
      I32EnumAttrCase<"HalfAwayFromZero", 1>,
      I32EnumAttrCase<"TowardsZero", 2>,
      I32EnumAttrCase<"Down", 3>,
      I32EnumAttrCase<"Up", 4>,
      I32EnumAttrCase<"HalfUp", 5>,
      I32EnumAttrCase<"HalfDown", 6>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tpu_mlir::sg2260";
}

def SG2260_ArithType: I32EnumAttr<"ArithType", "",
    [
      I32EnumAttrCase<"add", 0>,
      I32EnumAttrCase<"sub", 1>,
      I32EnumAttrCase<"mul", 2>,
      I32EnumAttrCase<"div", 3>,
      I32EnumAttrCase<"max", 4>,
      I32EnumAttrCase<"min", 5>,
      I32EnumAttrCase<"AND", 6>,
      I32EnumAttrCase<"OR",  7>,
      I32EnumAttrCase<"XOR", 8>,
      I32EnumAttrCase<"sgt", 9>,
      I32EnumAttrCase<"seq", 10>,
      I32EnumAttrCase<"slt", 11>,
      I32EnumAttrCase<"abs", 12>,
      I32EnumAttrCase<"neg", 13>
  ]>{
  let cppNamespace = "::tpu_mlir::sg2260";
}

#endif // SG2260_ENUMS
