#ifndef SOPHGO_TPU_BASE
#define SOPHGO_TPU_BASE

include "mlir/IR/OpBase.td"

def Tpu_Dialect : Dialect {
  let name = "tpu";
  let summary = "A tpu dialect for the SOPHGO AI chips";
  let cppNamespace = "::sophgo::tpu";
}

#endif
