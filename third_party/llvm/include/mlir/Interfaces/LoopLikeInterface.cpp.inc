/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Returns true if the given value is defined outside of the loop.
/// A sensible implementation could be to check whether the value's defining
/// operation lies outside of the loops body region. If the loop uses
/// explicit capture of dependencies, an implementation could check whether
/// the value corresponds to a captured dependency.
bool mlir::LoopLikeOpInterface::isDefinedOutsideOfLoop(::mlir::Value  value) {
      return getImpl()->isDefinedOutsideOfLoop(getImpl(), getOperation(), value);
  }
/// Returns the region that makes up the body of the loop and should be
/// inspected for loop-invariant operations.
::mlir::Region &mlir::LoopLikeOpInterface::getLoopBody() {
      return getImpl()->getLoopBody(getImpl(), getOperation());
  }
/// Moves the given loop-invariant operation out of the loop.
void mlir::LoopLikeOpInterface::moveOutOfLoop(::mlir::Operation * op) {
      return getImpl()->moveOutOfLoop(getImpl(), getOperation(), op);
  }
/// If there is a single induction variable return it, otherwise return
/// std::nullopt.
::std::optional<::mlir::Value> mlir::LoopLikeOpInterface::getSingleInductionVar() {
      return getImpl()->getSingleInductionVar(getImpl(), getOperation());
  }
/// Return the single lower bound value or attribute if it exists, otherwise
/// return std::nullopt.
::std::optional<::mlir::OpFoldResult> mlir::LoopLikeOpInterface::getSingleLowerBound() {
      return getImpl()->getSingleLowerBound(getImpl(), getOperation());
  }
/// Return the single step value or attribute if it exists, otherwise
/// return std::nullopt.
::std::optional<::mlir::OpFoldResult> mlir::LoopLikeOpInterface::getSingleStep() {
      return getImpl()->getSingleStep(getImpl(), getOperation());
  }
/// Return the single upper bound value or attribute if it exists, otherwise
/// return std::nullopt.
::std::optional<::mlir::OpFoldResult> mlir::LoopLikeOpInterface::getSingleUpperBound() {
      return getImpl()->getSingleUpperBound(getImpl(), getOperation());
  }
