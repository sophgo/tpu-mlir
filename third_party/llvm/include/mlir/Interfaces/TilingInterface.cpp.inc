/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Method to generate a tensor initalized with the identity value of the
/// operation reduction. The tensor shape is equal to operation result
/// shape with new dimension for each non zero tile size.
FailureOr<Operation*> mlir::PartialReductionOpInterface::generateInitialTensorForPartialReduction(OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim) {
      return getImpl()->generateInitialTensorForPartialReduction(getImpl(), getOperation(), b, loc, sizes, reductionDim);
  }
/// Method to generate a tiled version of the operation where the tiled
/// reduction dimension are converted to parallel dimensions with a size
/// less or equal to the tile size. This is meant to be used with
/// `mergeReductions` method which will combine the partial reductions.
Operation*mlir::PartialReductionOpInterface::tileToPartialReduction(OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) {
      return getImpl()->tileToPartialReduction(getImpl(), getOperation(), b, loc, init, offsets, sizes, reductionDims);
  }
/// Method to merge partial reductions for an operation that has been
/// tiled along the reduction dimensions. This will only apply the
/// reduction the operation.
Operation*mlir::PartialReductionOpInterface::mergeReductions(OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim) {
      return getImpl()->mergeReductions(getImpl(), getOperation(), b, loc, partialReduce, reductionDim);
  }
/// Returns a list of iterator types that describe the number of loops.
SmallVector<utils::IteratorType> mlir::TilingInterface::getLoopIteratorTypes() {
      return getImpl()->getLoopIteratorTypes(getImpl(), getOperation());
  }
/// Returns a list of ranges that describe the loop bounds and
/// step for the loops of the operation.
SmallVector<Range> mlir::TilingInterface::getIterationDomain(OpBuilder & b) {
      return getImpl()->getIterationDomain(getImpl(), getOperation(), b);
  }
/// Method to generate the tiled implementation of an operation.
/// 
/// The iteration space of the operation is returned by
/// `getIterationDomain`. The caller provides the information of the
/// tile within this iteration space whose implementation the
/// caller needs.
/// - `offsets` provides the offset of the tile in the coordinate system
///   of the original iteration space, i.e., if an iteration space
///   dimension had non-zero offset, it must be included in the offset
///   provided here (as opposed to zero-based offset "relative" to the
///   iteration space).
/// - `sizes` provides the size of the tile.
/// 
/// The method returns the operation that is the tiled
/// implementation.
SmallVector<Operation *> mlir::TilingInterface::getTiledImplementation(OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
      return getImpl()->getTiledImplementation(getImpl(), getOperation(), b, offsets, sizes);
  }
/// Method to return the position of the result tile computed by the tiled operation.
/// 
/// Specifies what tile of the result of the original tensor is computed
/// by the tiled implementation. Expects the same `offsets` and `sizes` as
/// used to obtain the tiled implementation of the operation.
LogicalResult mlir::TilingInterface::getResultTilePosition(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes) {
      return getImpl()->getResultTilePosition(getImpl(), getOperation(), b, resultNumber, offsets, sizes, resultOffsets, resultSizes);
  }
/// Method to generate the code that produces a tile of the result.
/// 
/// Generates the IR that computes the tile of a result of the
/// operation.  The `offsets` and `sizes` describe the tile of
/// the output required. This is different from
/// `getTiledImplementation` which generates the tiled
/// implementation of the operation given a tile of the
/// iteration space. This method generates a tiled
/// implementation of the operation based on the tile of the
/// result required. This method enables fusion by using tile
/// and fuse. The method returns failure if the operation can't be
/// tiled to generate the result tile. In practical terms this
/// implies it cannot be tiled and fused with its consumers.
/// 
/// - `offsets` provides the offset of the tile in the coordinate system
///   of the original iteration space, i.e., if an iteration space
///   dimension had non-zero offset, it must be included in the offset
///   provided here (as opposed to zero-based offset "relative" to the
///   iteration space).
/// - `sizes` provides the size of the tile.
FailureOr<Value> mlir::TilingInterface::generateResultTileValue(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
      return getImpl()->generateResultTileValue(getImpl(), getOperation(), b, resultNumber, offsets, sizes);
  }
/// Generates the scalar implementation of the operation.
/// 
/// Given the list `ivs` that represent points in the iteration space
/// (as specified by `getIterationDomain()`) returns the scalar operations
/// that represent the computation at that point in the iteration space.
/// This method is typically used as the "exit path", i.e. once all
/// transformations are done, this method can be used to lower to scalar
/// code that can then be lowered to LLVM or SPIR-V dialects.
LogicalResult mlir::TilingInterface::generateScalarImplementation(OpBuilder & b, Location  loc, ValueRange  ivs) {
      return getImpl()->generateScalarImplementation(getImpl(), getOperation(), b, loc, ivs);
  }
