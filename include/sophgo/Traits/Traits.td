
#ifndef SOPHGO_TRAITS
#define SOPHGO_TRAITS

include "mlir/IR/OpBase.td"

class SophgoOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::sophgo::trait";
}

def SupportFuseRelu : SophgoOpTrait<"SupportFuseRelu">;
def HasCommonAttributes : SophgoOpTrait<"HasCommonAttributes">;
def TpuTypeRestrict: SophgoOpTrait<"TpuTypeRestrict">;
#endif
