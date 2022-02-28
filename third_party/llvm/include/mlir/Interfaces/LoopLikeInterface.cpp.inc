/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool mlir::LoopLikeOpInterface::isDefinedOutsideOfLoop(::mlir::Value  value) {
      return getImpl()->isDefinedOutsideOfLoop(getImpl(), getOperation(), value);
  }
::mlir::Region &mlir::LoopLikeOpInterface::getLoopBody() {
      return getImpl()->getLoopBody(getImpl(), getOperation());
  }
::mlir::LogicalResult mlir::LoopLikeOpInterface::moveOutOfLoop(::mlir::ArrayRef<::mlir::Operation *> ops) {
      return getImpl()->moveOutOfLoop(getImpl(), getOperation(), ops);
  }
