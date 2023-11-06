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
include "tpu_mlir/Dialect/SG2260/IR/SG2260.td"

class SG2260_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SG2260_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

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
def ShortMatrix2RegDefProperties : RegProperty<"ShortMatrix2RegDef">;


#endif
