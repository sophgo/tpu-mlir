//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_INDEXINGMAPSINTERFACE
#define TPU_MLIR_INDEXINGMAPSINTERFACE

include "mlir/IR/OpBase.td"

def IndexingMapsInterface : OpInterface<"IndexingMapsInterface"> {
  let description = [{}];
  let cppNamespace = "::tpu_mlir";
  let methods = [
    InterfaceMethod<
        /*desc=*/[{This interface is like the methold in
                       linalg(`static_indexing_maps`),
                   but dynamic generated with the current
                       operation.Only provide parallel information by now.}],
        /*retType=*/"mlir::ArrayAttr",
        /*methodName=*/"getIndexingMaps",
        /*args=*/(ins), [{}]>,
  ];
}
#endif // TPU_MLIR_INDEXINGMAPSINTERFACE
