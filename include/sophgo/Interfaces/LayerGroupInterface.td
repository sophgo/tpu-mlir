#ifndef SOPHGO_LAYERGROUPINTERFACE
#define SOPHGO_LAYERGROUPINTERFACE

include "mlir/IR/OpBase.td"

def LayerGroupInterface : OpInterface<"LayerGroupInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"Verify",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"BackwardH",
        /*args=*/(ins "int64_t&":$in_idx, "int64_t&":$in_slice, "int64_t":$out_idx, "int64_t":$out_slice),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_idx = out_idx;
          in_slice = out_slice;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"BackwardN",
        /*args=*/(ins "int64_t&":$in_idx, "int64_t&":$in_slice, "int64_t":$out_idx, "int64_t":$out_slice),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          in_idx = out_idx;
          in_slice = out_slice;
          return mlir::success();
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"getBufferSize",
        /*args=*/(ins "int64_t":$out_n, "int64_t":$out_c, "int64_t":$out_h, "int64_t":$out_w, "int64_t":$out_lmem_bytes),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return 0;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"sophgo::group_info_t",
        /*methodName=*/"getGroupInfo",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return LayerGroupInterface::getGroupInfo(this->getOperation(), n_step, h_step);
        }]
      >,
  ];
  let extraClassDeclaration = [{
  static constexpr llvm::StringRef kLayerGroupAttrName = "group_info";
  static void fixSlice(int64_t &in_idx, int64_t &in_slice, int64_t in_length);
  static sophgo::group_info_t getGroupInfo(mlir::Operation * op, int64_t n_step = 0, int64_t h_step = 0);
  static sophgo::group_info_t getGroupInfo(mlir::Value, int64_t n_step = 0, int64_t h_step = 0);
  }];
}
#endif // SOPHGO_LAYERGROUPINTERFACE
