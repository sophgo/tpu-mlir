//===- ConversionUtils.h - Helper functions for tosa conversion -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions for TOSA lowering
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_TOSA_UTILS_COVERSION_UTILS_H_
#define DIALECT_TOSA_UTILS_COVERSION_UTILS_H_

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace tosa {

// Creates a SmallVector of Stringrefs for N parallel loops
SmallVector<utils::IteratorType>
getNParallelLoopsAttrs(unsigned nParallelLoops);

// Takes a vector of values and condenses them to a vector with no gaps.
SmallVector<Value> condenseValues(const SmallVector<Value> &values);

// Takes the parameters for a clamp and turns it into a series of ops for float
// inputs.
Value clampFloatHelper(Location loc, Value arg, Value min, Value max,
                       OpBuilder &rewriter);

// Takes the parameters for a clamp and turns it into a series of ops for
// integer inputs.
Value clampIntHelper(Location loc, Value arg, Value min, Value max,
                     OpBuilder &rewriter);

// Determines whether the integer value falls witin the range of integer type.
bool validIntegerRange(IntegerType ty, int64_t value);

// Returns the values in an attribute as an array of values.
template <typename T>
void getValuesFromIntArrayAttribute(ArrayAttr attr,
                                    SmallVector<T> &arrayValues) {
  for (Attribute val : attr.getValue()) {
    arrayValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

} // namespace tosa
} // namespace mlir

#endif // DIALECT_TOSA_UTILS_COVERSION_UTILS_H_
