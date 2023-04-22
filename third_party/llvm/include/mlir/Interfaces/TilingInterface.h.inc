/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class TilingInterface;
namespace detail {
struct TilingInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    SmallVector<utils::IteratorType> (*getLoopIteratorTypes)(const Concept *impl, ::mlir::Operation *);
    SmallVector<Range> (*getIterationDomain)(const Concept *impl, ::mlir::Operation *, OpBuilder &);
    FailureOr<TilingResult> (*getTiledImplementation)(const Concept *impl, ::mlir::Operation *, OpBuilder &, ArrayRef<OpFoldResult> , ArrayRef<OpFoldResult> );
    LogicalResult (*getResultTilePosition)(const Concept *impl, ::mlir::Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult> , ArrayRef<OpFoldResult> , SmallVector<OpFoldResult> &, SmallVector<OpFoldResult> &);
    FailureOr<TilingResult> (*generateResultTileValue)(const Concept *impl, ::mlir::Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult>, ArrayRef<OpFoldResult>);
    LogicalResult (*generateScalarImplementation)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location , ValueRange );
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::TilingInterface;
    Model() : Concept{getLoopIteratorTypes, getIterationDomain, getTiledImplementation, getResultTilePosition, generateResultTileValue, generateScalarImplementation} {}

    static inline SmallVector<utils::IteratorType> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Range> getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline FailureOr<TilingResult> getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
    static inline LogicalResult getResultTilePosition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes);
    static inline FailureOr<TilingResult> generateResultTileValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
    static inline LogicalResult generateScalarImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange  ivs);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::TilingInterface;
    FallbackModel() : Concept{getLoopIteratorTypes, getIterationDomain, getTiledImplementation, getResultTilePosition, generateResultTileValue, generateScalarImplementation} {}

    static inline SmallVector<utils::IteratorType> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Range> getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline FailureOr<TilingResult> getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
    static inline LogicalResult getResultTilePosition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes);
    static inline FailureOr<TilingResult> generateResultTileValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
    static inline LogicalResult generateScalarImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange  ivs);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    SmallVector<utils::IteratorType> getLoopIteratorTypes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<Range> getIterationDomain(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const;
    FailureOr<TilingResult> getTiledImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const;
    LogicalResult getResultTilePosition(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets, SmallVector<OpFoldResult> &resultSizes) const;
    FailureOr<TilingResult> generateResultTileValue(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const;
    LogicalResult generateScalarImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ValueRange ivs) const;
  };
};template <typename ConcreteOp>
struct TilingInterfaceTrait;

} // namespace detail
class TilingInterface : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TilingInterfaceTrait<ConcreteOp> {};
  /// Returns a list of iterator types that describe the number of loops.
  SmallVector<utils::IteratorType> getLoopIteratorTypes();
  /// Returns a list of ranges that describe the loop bounds and
  /// step for the loops of the operation.
  SmallVector<Range> getIterationDomain(OpBuilder & b);
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
  FailureOr<TilingResult> getTiledImplementation(OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes);
  /// Method to return the position of the result tile computed by the tiled operation.
  /// 
  /// Specifies what tile of the result of the original tensor is computed
  /// by the tiled implementation. Expects the same `offsets` and `sizes` as
  /// used to obtain the tiled implementation of the operation.
  LogicalResult getResultTilePosition(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes);
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
  FailureOr<TilingResult> generateResultTileValue(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes);
  /// Generates the scalar implementation of the operation.
  /// 
  /// Given the list `ivs` that represent points in the iteration space
  /// (as specified by `getIterationDomain()`) returns the scalar operations
  /// that represent the computation at that point in the iteration space.
  /// This method is typically used as the "exit path", i.e. once all
  /// transformations are done, this method can be used to lower to scalar
  /// code that can then be lowered to LLVM or SPIR-V dialects.
  LogicalResult generateScalarImplementation(OpBuilder & b, Location  loc, ValueRange  ivs);
};
namespace detail {
  template <typename ConcreteOp>
  struct TilingInterfaceTrait : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a list of iterator types that describe the number of loops.
    SmallVector<utils::IteratorType> getLoopIteratorTypes() {
      return {};
    }
    /// Returns a list of ranges that describe the loop bounds and
    /// step for the loops of the operation.
    SmallVector<Range> getIterationDomain(OpBuilder & b) {
      return {};
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
    FailureOr<TilingResult> getTiledImplementation(OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
      return {};
    }
    /// Method to return the position of the result tile computed by the tiled operation.
    /// 
    /// Specifies what tile of the result of the original tensor is computed
    /// by the tiled implementation. Expects the same `offsets` and `sizes` as
    /// used to obtain the tiled implementation of the operation.
    LogicalResult getResultTilePosition(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes) {
      return failure();
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
    FailureOr<TilingResult> generateResultTileValue(OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
      return failure();
    }
    /// Generates the scalar implementation of the operation.
    /// 
    /// Given the list `ivs` that represent points in the iteration space
    /// (as specified by `getIterationDomain()`) returns the scalar operations
    /// that represent the computation at that point in the iteration space.
    /// This method is typically used as the "exit path", i.e. once all
    /// transformations are done, this method can be used to lower to scalar
    /// code that can then be lowered to LLVM or SPIR-V dialects.
    LogicalResult generateScalarImplementation(OpBuilder & b, Location  loc, ValueRange  ivs) {
      return failure();
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
class PartialReductionOpInterface;
namespace detail {
struct PartialReductionOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    FailureOr<Operation*> (*generateInitialTensorForPartialReduction)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location , ArrayRef<OpFoldResult>, ArrayRef<int>);
    Operation*(*tileToPartialReduction)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location , ValueRange, ArrayRef<OpFoldResult>, ArrayRef<OpFoldResult>, ArrayRef<int>);
    Operation*(*mergeReductions)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location , ValueRange, ArrayRef<int>);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::PartialReductionOpInterface;
    Model() : Concept{generateInitialTensorForPartialReduction, tileToPartialReduction, mergeReductions} {}

    static inline FailureOr<Operation*> generateInitialTensorForPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim);
    static inline Operation*tileToPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims);
    static inline Operation*mergeReductions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::PartialReductionOpInterface;
    FallbackModel() : Concept{generateInitialTensorForPartialReduction, tileToPartialReduction, mergeReductions} {}

    static inline FailureOr<Operation*> generateInitialTensorForPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim);
    static inline Operation*tileToPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims);
    static inline Operation*mergeReductions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    FailureOr<Operation*> generateInitialTensorForPartialReduction(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim) const;
    Operation*tileToPartialReduction(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) const;
    Operation*mergeReductions(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ValueRange partialReduce, ArrayRef<int> reductionDim) const;
  };
};template <typename ConcreteOp>
struct PartialReductionOpInterfaceTrait;

} // namespace detail
class PartialReductionOpInterface : public ::mlir::OpInterface<PartialReductionOpInterface, detail::PartialReductionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<PartialReductionOpInterface, detail::PartialReductionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::PartialReductionOpInterfaceTrait<ConcreteOp> {};
  /// Method to generate a tensor initalized with the identity value of the
  /// operation reduction. The tensor shape is equal to operation result
  /// shape with new dimension for each non zero tile size.
  FailureOr<Operation*> generateInitialTensorForPartialReduction(OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim);
  /// Method to generate a tiled version of the operation where the tiled
  /// reduction dimension are converted to parallel dimensions with a size
  /// less or equal to the tile size. This is meant to be used with
  /// `mergeReductions` method which will combine the partial reductions.
  Operation*tileToPartialReduction(OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims);
  /// Method to merge partial reductions for an operation that has been
  /// tiled along the reduction dimensions. This will only apply the
  /// reduction the operation.
  Operation*mergeReductions(OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim);
};
namespace detail {
  template <typename ConcreteOp>
  struct PartialReductionOpInterfaceTrait : public ::mlir::OpInterface<PartialReductionOpInterface, detail::PartialReductionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Method to generate a tensor initalized with the identity value of the
    /// operation reduction. The tensor shape is equal to operation result
    /// shape with new dimension for each non zero tile size.
    FailureOr<Operation*> generateInitialTensorForPartialReduction(OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim) {
      return failure();
    }
    /// Method to generate a tiled version of the operation where the tiled
    /// reduction dimension are converted to parallel dimensions with a size
    /// less or equal to the tile size. This is meant to be used with
    /// `mergeReductions` method which will combine the partial reductions.
    Operation*tileToPartialReduction(OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) {
      return nullptr;
    }
    /// Method to merge partial reductions for an operation that has been
    /// tiled along the reduction dimensions. This will only apply the
    /// reduction the operation.
    Operation*mergeReductions(OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim) {
      return nullptr;
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
SmallVector<utils::IteratorType> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopIteratorTypes();
}
template<typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIterationDomain(b);
}
template<typename ConcreteOp>
FailureOr<TilingResult> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiledImplementation(b, offsets, sizes);
}
template<typename ConcreteOp>
LogicalResult detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getResultTilePosition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultTilePosition(b, resultNumber, offsets, sizes, resultOffsets, resultSizes);
}
template<typename ConcreteOp>
FailureOr<TilingResult> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::generateResultTileValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).generateResultTileValue(b, resultNumber, offsets, sizes);
}
template<typename ConcreteOp>
LogicalResult detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::generateScalarImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange  ivs) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).generateScalarImplementation(b, loc, ivs);
}
template<typename ConcreteOp>
SmallVector<utils::IteratorType> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopIteratorTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return static_cast<const ConcreteOp *>(impl)->getIterationDomain(tablegen_opaque_val, b);
}
template<typename ConcreteOp>
FailureOr<TilingResult> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes) {
  return static_cast<const ConcreteOp *>(impl)->getTiledImplementation(tablegen_opaque_val, b, offsets, sizes);
}
template<typename ConcreteOp>
LogicalResult detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getResultTilePosition(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, SmallVector<OpFoldResult> & resultOffsets, SmallVector<OpFoldResult> & resultSizes) {
  return static_cast<const ConcreteOp *>(impl)->getResultTilePosition(tablegen_opaque_val, b, resultNumber, offsets, sizes, resultOffsets, resultSizes);
}
template<typename ConcreteOp>
FailureOr<TilingResult> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::generateResultTileValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) {
  return static_cast<const ConcreteOp *>(impl)->generateResultTileValue(tablegen_opaque_val, b, resultNumber, offsets, sizes);
}
template<typename ConcreteOp>
LogicalResult detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::generateScalarImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange  ivs) {
  return static_cast<const ConcreteOp *>(impl)->generateScalarImplementation(tablegen_opaque_val, b, loc, ivs);
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<utils::IteratorType> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopIteratorTypes(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIterationDomain(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
FailureOr<TilingResult> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiledImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getResultTilePosition(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets, SmallVector<OpFoldResult> &resultSizes) const {
return failure();
}
template<typename ConcreteModel, typename ConcreteOp>
FailureOr<TilingResult> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::generateResultTileValue(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes) const {
return failure();
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::generateScalarImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ValueRange ivs) const {
return failure();
}
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
FailureOr<Operation*> detail::PartialReductionOpInterfaceInterfaceTraits::Model<ConcreteOp>::generateInitialTensorForPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).generateInitialTensorForPartialReduction(b, loc, sizes, reductionDim);
}
template<typename ConcreteOp>
Operation*detail::PartialReductionOpInterfaceInterfaceTraits::Model<ConcreteOp>::tileToPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).tileToPartialReduction(b, loc, init, offsets, sizes, reductionDims);
}
template<typename ConcreteOp>
Operation*detail::PartialReductionOpInterfaceInterfaceTraits::Model<ConcreteOp>::mergeReductions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).mergeReductions(b, loc, partialReduce, reductionDim);
}
template<typename ConcreteOp>
FailureOr<Operation*> detail::PartialReductionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::generateInitialTensorForPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim) {
  return static_cast<const ConcreteOp *>(impl)->generateInitialTensorForPartialReduction(tablegen_opaque_val, b, loc, sizes, reductionDim);
}
template<typename ConcreteOp>
Operation*detail::PartialReductionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::tileToPartialReduction(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) {
  return static_cast<const ConcreteOp *>(impl)->tileToPartialReduction(tablegen_opaque_val, b, loc, init, offsets, sizes, reductionDims);
}
template<typename ConcreteOp>
Operation*detail::PartialReductionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::mergeReductions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location  loc, ValueRange partialReduce, ArrayRef<int> reductionDim) {
  return static_cast<const ConcreteOp *>(impl)->mergeReductions(tablegen_opaque_val, b, loc, partialReduce, reductionDim);
}
template<typename ConcreteModel, typename ConcreteOp>
FailureOr<Operation*> detail::PartialReductionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::generateInitialTensorForPartialReduction(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDim) const {
return failure();
}
template<typename ConcreteModel, typename ConcreteOp>
Operation*detail::PartialReductionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::tileToPartialReduction(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ValueRange init, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims) const {
return nullptr;
}
template<typename ConcreteModel, typename ConcreteOp>
Operation*detail::PartialReductionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::mergeReductions(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, Location loc, ValueRange partialReduce, ArrayRef<int> reductionDim) const {
return nullptr;
}
} // namespace mlir
