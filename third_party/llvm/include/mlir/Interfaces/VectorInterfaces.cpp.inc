/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Return the `in_bounds` attribute name.
::mlir::StringRef mlir::VectorTransferOpInterface::getInBoundsAttrStrName() {
      return getImpl()->getInBoundsAttrStrName();
  }
/// Return the `permutation_map` attribute name.
::mlir::StringRef mlir::VectorTransferOpInterface::getPermutationMapAttrStrName() {
      return getImpl()->getPermutationMapAttrStrName();
  }
/// Return `true` if dimension `dim` is in-bounds. Return `false`
///                 otherwise.
bool mlir::VectorTransferOpInterface::isDimInBounds(unsigned dim) {
      return getImpl()->isDimInBounds(getImpl(), getOperation(), dim);
  }
/// Return the memref or ranked tensor operand.
::mlir::Value mlir::VectorTransferOpInterface::source() {
      return getImpl()->source(getImpl(), getOperation());
  }
/// Return the vector operand or result.
::mlir::Value mlir::VectorTransferOpInterface::vector() {
      return getImpl()->vector(getImpl(), getOperation());
  }
/// Return the indices operands.
::mlir::ValueRange mlir::VectorTransferOpInterface::indices() {
      return getImpl()->indices(getImpl(), getOperation());
  }
/// Return the permutation map.
::mlir::AffineMap mlir::VectorTransferOpInterface::permutation_map() {
      return getImpl()->permutation_map(getImpl(), getOperation());
  }
/// Returns true if the specified dimension is a broadcast.
bool mlir::VectorTransferOpInterface::isBroadcastDim(unsigned idx) {
      return getImpl()->isBroadcastDim(getImpl(), getOperation(), idx);
  }
/// Returns true if at least one of the dimensions in the
///                  permutation map is a broadcast.
bool mlir::VectorTransferOpInterface::hasBroadcastDim() {
      return getImpl()->hasBroadcastDim(getImpl(), getOperation());
  }
/// Return the `in_bounds` boolean ArrayAttr.
::std::optional<::mlir::ArrayAttr> mlir::VectorTransferOpInterface::in_bounds() {
      return getImpl()->in_bounds(getImpl(), getOperation());
  }
/// Return the ShapedType.
::mlir::ShapedType mlir::VectorTransferOpInterface::getShapedType() {
      return getImpl()->getShapedType(getImpl(), getOperation());
  }
/// Return the VectorType.
::mlir::VectorType mlir::VectorTransferOpInterface::getVectorType() {
      return getImpl()->getVectorType(getImpl(), getOperation());
  }
/// Return the mask operand if the op has a mask. Otherwise, return a empty value.
Value mlir::VectorTransferOpInterface::getMask() {
      return getImpl()->getMask(getImpl(), getOperation());
  }
/// Return the mask type if the op has a mask. Otherwise, return an empty VectorType.
::mlir::VectorType mlir::VectorTransferOpInterface::getMaskType() {
      return getImpl()->getMaskType(getImpl(), getOperation());
  }
/// Return the number of dimensions that participate in the
///                  permutation map.
unsigned mlir::VectorTransferOpInterface::getTransferRank() {
      return getImpl()->getTransferRank(getImpl(), getOperation());
  }
/// Return the number of leading shaped dimensions that do not
///                  participate in the permutation map.
unsigned mlir::VectorTransferOpInterface::getLeadingShapedRank() {
      return getImpl()->getLeadingShapedRank(getImpl(), getOperation());
  }
/// Returns true if at least one of the dimensions may be
///                  out-of-bounds.
bool mlir::VectorTransferOpInterface::hasOutOfBoundsDim() {
      return getImpl()->hasOutOfBoundsDim(getImpl(), getOperation());
  }
/// Helper function to account for the fact that `permutationMap` results and
/// `op.indices` sizes may not match and may not be aligned. The first
/// `getLeadingShapedRank()` indices may just be indexed and not
/// transferred from/into the vector.
/// For example:
/// ```
///    vector.transfer %0[%i, %j, %k, %c0] :
///      memref<?x?x?x?xf32>, vector<2x4xf32>
/// ```
/// with `permutation_map = (d0, d1, d2, d3) -> (d2, d3)`.
/// Provide a zip function to coiterate on 2 running indices: `resultIdx` and
/// `indicesIdx` which accounts for this misalignment.
void mlir::VectorTransferOpInterface::zipResultAndIndexing(::llvm::function_ref<void(int64_t, int64_t)> fun) {
      return getImpl()->zipResultAndIndexing(getImpl(), getOperation(), fun);
  }
/// Return an upper-bound shape accessed by the transfer op within the
/// tensor/memref operand.
/// For example:
/// ```
///   vector.transfer %w0[%i, %j] {
///     permutation_map = affine_map<(d0, d1) -> (d1, d0, 0)>} :
///     tensor<?x?xf32>, vector<4x2x6xf32>
/// ```
/// returns a shape [2, 4].
SmallVector<int64_t> mlir::VectorTransferOpInterface::getTransferChunkAccessed() {
      return getImpl()->getTransferChunkAccessed(getImpl(), getOperation());
  }
/// Return the shape ratio of unrolling to the target vector shape
/// `targetShape`. Return `std::nullopt` if the op cannot be unrolled to the target
/// vector shape.
::std::optional<::llvm::SmallVector<int64_t, 4>> mlir::VectorUnrollOpInterface::getShapeForUnroll() {
      return getImpl()->getShapeForUnroll(getImpl(), getOperation());
  }
