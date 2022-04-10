#pragma once

#include "sophgo/Support/Dnnl/Conv.h"
#include "sophgo/Support/Dnnl/MatMul.h"
#include "sophgo/Support/Dnnl/Pool.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

namespace sophgo {

dnnl::memory::data_type getDnnlType(mlir::Value v);

}
