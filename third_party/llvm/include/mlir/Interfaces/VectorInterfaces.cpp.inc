/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

::mlir::StringRef mlir::VectorTransferOpInterface::getInBoundsAttrStrName() {
      return getImpl()->getInBoundsAttrStrName();
  }
::mlir::StringRef mlir::VectorTransferOpInterface::getPermutationMapAttrStrName() {
      return getImpl()->getPermutationMapAttrStrName();
  }
bool mlir::VectorTransferOpInterface::isDimInBounds(unsigned dim) {
      return getImpl()->isDimInBounds(getImpl(), getOperation(), dim);
  }
::mlir::Value mlir::VectorTransferOpInterface::source() {
      return getImpl()->source(getImpl(), getOperation());
  }
::mlir::Value mlir::VectorTransferOpInterface::vector() {
      return getImpl()->vector(getImpl(), getOperation());
  }
::mlir::ValueRange mlir::VectorTransferOpInterface::indices() {
      return getImpl()->indices(getImpl(), getOperation());
  }
::mlir::AffineMap mlir::VectorTransferOpInterface::permutation_map() {
      return getImpl()->permutation_map(getImpl(), getOperation());
  }
bool mlir::VectorTransferOpInterface::isBroadcastDim(unsigned idx) {
      return getImpl()->isBroadcastDim(getImpl(), getOperation(), idx);
  }
bool mlir::VectorTransferOpInterface::hasBroadcastDim() {
      return getImpl()->hasBroadcastDim(getImpl(), getOperation());
  }
::mlir::Optional<::mlir::ArrayAttr> mlir::VectorTransferOpInterface::in_bounds() {
      return getImpl()->in_bounds(getImpl(), getOperation());
  }
::mlir::ShapedType mlir::VectorTransferOpInterface::getShapedType() {
      return getImpl()->getShapedType(getImpl(), getOperation());
  }
::mlir::VectorType mlir::VectorTransferOpInterface::getVectorType() {
      return getImpl()->getVectorType(getImpl(), getOperation());
  }
::mlir::VectorType mlir::VectorTransferOpInterface::getMaskType() {
      return getImpl()->getMaskType(getImpl(), getOperation());
  }
unsigned mlir::VectorTransferOpInterface::getTransferRank() {
      return getImpl()->getTransferRank(getImpl(), getOperation());
  }
unsigned mlir::VectorTransferOpInterface::getLeadingShapedRank() {
      return getImpl()->getLeadingShapedRank(getImpl(), getOperation());
  }
bool mlir::VectorTransferOpInterface::hasOutOfBoundsDim() {
      return getImpl()->hasOutOfBoundsDim(getImpl(), getOperation());
  }
void mlir::VectorTransferOpInterface::zipResultAndIndexing(::llvm::function_ref<void(int64_t, int64_t)> fun) {
      return getImpl()->zipResultAndIndexing(getImpl(), getOperation(), fun);
  }
::mlir::Optional<::mlir::SmallVector<int64_t, 4>> mlir::VectorUnrollOpInterface::getShapeForUnroll() {
      return getImpl()->getShapeForUnroll(getImpl(), getOperation());
  }
