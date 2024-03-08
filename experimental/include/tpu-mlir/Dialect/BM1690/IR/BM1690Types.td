//===-- BM1690Types.td - BM1690 types  definitions *- tablegen -*----------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.

//
//===----------------------------------------------------------------------===//
#ifndef TPUMLIR_DIALECT_BM1690_IR_BM1690TYPES
#define TPUMLIR_DIALECT_BM1690_IR_BM1690TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "BM1690.td"
include "BM1690Enums.td"


class BM1690_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<BM1690_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IteratorTypeEnum : EnumAttr<BM1690_Dialect, IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}
def IteratorTypeArrayAttr : TypedArrayAttrBase<IteratorTypeEnum,
  "Iterator type should be an enum.">;

def BM1690_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def BM1690_IntArrayAttr3 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>;
def BM1690_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;

class InsID<string name, string typeMnemonic>
    : BM1690_Type<name, typeMnemonic> {
  let parameters = (ins "unsigned":$id);
  let assemblyFormat = "`<` $id `>`";
}

def BM1690_TIUIdType : InsID<"TIUId", "tiu.id"> {
  let summary = "TIU identifier.";
  let description = [{
    The ID used by TIU insturcion.
  }];
}

def BM1690_DMAIdType : InsID<"DMAId", "dma.id"> {
  let summary = "DMA identifier.";
  let description = [{
    The ID used by DMA insturcion.
  }];
}

def BM1690_PaddingModeAttr : EnumAttr<BM1690_Dialect, BM1690_PaddingMode, "Pad_Mode">;
def BM1690_RoundModeAttr : EnumAttr<BM1690_Dialect, BM1690_RoundMode, "round_mode">;

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
def ShortRQDQRegDefProperties : RegProperty<"ShortRQDQRegDef">;
def ShortPorDRegDefProperties : RegProperty<"ShortPorDRegDef">;
def ShortVCRegDefProperties : RegProperty<"ShortVCRegDef">;
def ShortCWBCRegDefProperties : RegProperty<"ShortCWBCRegDef">;
def SYSRegDefProperties : RegProperty<"SYSRegDef">;
def DMATensorRegDefProperties : RegProperty<"DMATensorRegDef">;
def GDMAMatrixRegDefProperties : RegProperty<"GDMAMatrixRegDef">;
def GDMAGeneralRegDefProperties : RegProperty<"GDMAGeneralRegDef">;
def GDMAGatherRegDefProperties : RegProperty<"GDMAGatherRegDef">;
def GDMAScatterRegDefProperties : RegProperty<"GDMAScatterRegDef">;
def ShortLINRegDefProperties : RegProperty<"ShortLINRegDef">;

#endif
