//===-- SG2260Types.td - SG2260 types  definitions *- tablegen -*----------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//
#ifndef TPUMLIR_DIALECT_SG2260_IR_SG2260TYPES
#define TPUMLIR_DIALECT_SG2260_IR_SG2260TYPES

include "mlir/IR/AttrTypeBase.td"
include "SG2260.td"
include "mlir/IR/EnumAttr.td"

class SG2260_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SG2260_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SG2260_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def SG2260_IntArrayAttr3 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>;
def SG2260_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;

class InsID<string name, string typeMnemonic>
    : SG2260_Type<name, typeMnemonic> {
  let parameters = (ins "unsigned":$id);
  let assemblyFormat = "`<` $id `>`";
}

def SG2260_TIUIdType : InsID<"TIUId", "tiu.id"> {
  let summary = "TIU identifier.";
  let description = [{
    The ID used by TIU insturcion.
  }];
}

def SG2260_DMAIdType : InsID<"DMAId", "dma.id"> {
  let summary = "DMA identifier.";
  let description = [{
    The ID used by DMA insturcion.
  }];
}

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
def SG2260_PaddingModeAttr : EnumAttr<SG2260_Dialect, SG2260_PaddingMode, "Pad_Mode">;

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
def SG2260_RoundModeAttr : EnumAttr<SG2260_Dialect, SG2260_RoundMode, "round_mode">;

class RegProperty<string name> : Property<name> {
  let convertToAttribute = [{ dummyPropertiesAsAttribute($_ctxt, $_storage) }];
  let convertFromAttribute = [{ return ::mlir::success(); }];
  let hashProperty = [{ 0 }];
  let writeToMlirBytecode = [{ (void)$_storage; }];
  let readFromMlirBytecode = [{
    (void)$_storage;
    return ::mlir::success();
  }];
}
def ShortMMRegDefProperties : RegProperty<"ShortMMRegDef">;
def ShortConvRegDefProperties : RegProperty<"ShortCONVRegDef">;
def ShortARRegDefProperties : RegProperty<"ShortARRegDef">;
def ShortSFURegDefProperties : RegProperty<"ShortSFURegDef">;
def ShortPorDRegDefProperties : RegProperty<"ShortPorDRegDef">;
def ShortVCRegDefProperties : RegProperty<"ShortVCRegDef">;
def ShortCWBCRegDefProperties : RegProperty<"ShortCWBCRegDef">;
def SYSRegDefProperties : RegProperty<"SYSRegDef">;
def DMATensorRegDefProperties : RegProperty<"DMATensorRegDef">;
def GDMAMatrixRegDefProperties : RegProperty<"GDMAMatrixRegDef">;
def GDMAGeneralRegDefProperties : RegProperty<"GDMAGeneralRegDef">;
def GDMAGatherRegDefProperties : RegProperty<"GDMAGatherRegDef">;
def GDMAScatterRegDefProperties : RegProperty<"GDMAScatterRegDef">;

#endif
