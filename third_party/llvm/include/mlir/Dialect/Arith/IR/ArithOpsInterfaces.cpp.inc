/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns a FastMathFlagsAttr attribute for the operation
FastMathFlagsAttr mlir::arith::ArithFastMathInterface::getFastMathFlagsAttr() {
      return getImpl()->getFastMathFlagsAttr(getImpl(), getOperation());
  }
/// Returns the name of the FastMathFlagsAttr attribute
///                          for the operation
StringRef mlir::arith::ArithFastMathInterface::getFastMathAttrName() {
      return getImpl()->getFastMathAttrName();
  }
