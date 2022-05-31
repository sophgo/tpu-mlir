//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

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
