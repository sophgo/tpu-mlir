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
::mlir::Optional<::mlir::Value> mlir::LoopLikeOpInterface::getSingleInductionVar() {
      return getImpl()->getSingleInductionVar(getImpl(), getOperation());
  }
::mlir::Optional<::mlir::OpFoldResult> mlir::LoopLikeOpInterface::getSingleLowerBound() {
      return getImpl()->getSingleLowerBound(getImpl(), getOperation());
  }
::mlir::Optional<::mlir::OpFoldResult> mlir::LoopLikeOpInterface::getSingleStep() {
      return getImpl()->getSingleStep(getImpl(), getOperation());
  }
