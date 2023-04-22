/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace affine {
class AffineApplyOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineDelinearizeIndexOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineForOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineIfOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineLoadOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineMaxOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineMinOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineParallelOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffinePrefetchOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineStoreOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineVectorLoadOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineVectorStoreOp;
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {
class AffineYieldOp;
} // namespace affine
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineApplyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineApplyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineApplyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineApplyOpGenericAdaptor : public detail::AffineApplyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineApplyOpGenericAdaptorBase;
public:
  AffineApplyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getMapOperands() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineApplyOpAdaptor : public AffineApplyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineApplyOpGenericAdaptor::AffineApplyOpGenericAdaptor;
  AffineApplyOpAdaptor(AffineApplyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineApplyOp : public ::mlir::Op<AffineApplyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineApplyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineApplyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.apply");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getMapOperands();
  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr);
  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap map, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<AffineExpr>  exprList, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Returns the affine map to be applied by this operation.
  AffineMap getAffineMap() { return getMap(); }

  /// Returns the affine value map computed from this operation.
  AffineValueMap getAffineValueMap();

  /// Returns true if the result of this operation can be used as dimension id
  /// in the region of the closest surrounding op with trait AffineScope.
  bool isValidDim();

  /// Returns true if the result of this operation can be used as dimension id
  /// within 'region', i.e., for all its uses with `region`.
  bool isValidDim(Region *region);

  /// Returns true if the result of this operation is a symbol in the region
  /// of the closest surrounding op that has the trait AffineScope.
  bool isValidSymbol();

  /// Returns true if the result of this operation is a symbol for all its
  /// uses in `region`.
  bool isValidSymbol(Region *region);

  /// Returns all dimension operands.
  ValueRange getDimOperands() {
    return OperandRange{getOperands().begin(),
                        getOperands().begin() + getMap().getNumDims()};
  }

  /// Returns all symbol operands.
  ValueRange getSymbolOperands() {
    return OperandRange{getOperands().begin() + getMap().getNumDims(),
                        getOperands().end()};
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineApplyOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineDelinearizeIndexOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineDelinearizeIndexOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineDelinearizeIndexOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AffineDelinearizeIndexOpGenericAdaptor : public detail::AffineDelinearizeIndexOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineDelinearizeIndexOpGenericAdaptorBase;
public:
  AffineDelinearizeIndexOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLinearIndex() {
    return (*getODSOperands(0).begin());
  }

  RangeT getBasis() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineDelinearizeIndexOpAdaptor : public AffineDelinearizeIndexOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineDelinearizeIndexOpGenericAdaptor::AffineDelinearizeIndexOpGenericAdaptor;
  AffineDelinearizeIndexOpAdaptor(AffineDelinearizeIndexOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineDelinearizeIndexOp : public ::mlir::Op<AffineDelinearizeIndexOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineDelinearizeIndexOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineDelinearizeIndexOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.delinearize_index");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getLinearIndex();
  ::mlir::Operation::operand_range getBasis();
  ::mlir::MutableOperandRange getLinearIndexMutable();
  ::mlir::MutableOperandRange getBasisMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getMultiIndex();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value linear_index, ArrayRef<OpFoldResult> basis);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange multi_index, ::mlir::Value linear_index, ::mlir::ValueRange basis);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineDelinearizeIndexOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineForOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineForOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineForOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AffineForOpGenericAdaptor : public detail::AffineForOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineForOpGenericAdaptorBase;
public:
  AffineForOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineForOpAdaptor : public AffineForOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineForOpGenericAdaptor::AffineForOpGenericAdaptor;
  AffineForOpAdaptor(AffineForOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineForOp : public ::mlir::Op<AffineForOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::LoopLikeOpInterface::Trait, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineForOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineForOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.for");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int64_t lowerBound, int64_t upperBound, int64_t step = 1, ValueRange iterArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lbOperands, AffineMap lbMap, ValueRange ubOperands, AffineMap ubMap, int64_t step = 1, ValueRange iterArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::mlir::Region &getLoopBody();
  ::std::optional<::mlir::Value> getSingleInductionVar();
  ::std::optional<::mlir::OpFoldResult> getSingleLowerBound();
  ::std::optional<::mlir::OpFoldResult> getSingleStep();
  ::std::optional<::mlir::OpFoldResult> getSingleUpperBound();
  ::mlir::OperandRange getSuccessorEntryOperands(::std::optional<unsigned> index);
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  /// Defining the function type we use for building the body of affine.for.
  using BodyBuilderFn =
      function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

  static StringRef getStepAttrStrName() { return "step"; }
  static StringRef getLowerBoundAttrStrName() { return "lower_bound"; }
  static StringRef getUpperBoundAttrStrName() { return "upper_bound"; }

  BlockArgument getInductionVar() { return getBody()->getArgument(0); }
  Block::BlockArgListType getRegionIterArgs() {
    return getBody()->getArguments().drop_front();
  }
  Operation::operand_range getIterOperands() {
    return getOperands().drop_front(getNumControlOperands());
  }

  // TODO: provide iterators for the lower and upper bound operands
  // if the current access via getLowerBound(), getUpperBound() is too slow.

  /// Returns operands for the lower bound map.
  operand_range getLowerBoundOperands();

  /// Returns operands for the upper bound map.
  operand_range getUpperBoundOperands();

  /// Returns operands for the lower and upper bound maps with the operands
  /// for the lower bound map in front of those for the upper bound map.
  operand_range getControlOperands();

  /// Returns information about the lower bound as a single object.
  AffineBound getLowerBound();

  /// Returns information about the upper bound as a single object.
  AffineBound getUpperBound();

  /// Returns loop step.
  int64_t getStep() {
    return (*this)->getAttr(getStepAttrStrName()).cast<IntegerAttr>().getInt();
  }

  /// Returns affine map for the lower bound.
  AffineMap getLowerBoundMap() { return getLowerBoundMapAttr().getValue(); }
  AffineMapAttr getLowerBoundMapAttr() {
    return (*this)->getAttr(getLowerBoundAttrStrName()).cast<AffineMapAttr>();
  }
  /// Returns affine map for the upper bound. The upper bound is exclusive.
  AffineMap getUpperBoundMap() { return getUpperBoundMapAttr().getValue(); }
  AffineMapAttr getUpperBoundMapAttr() {
    return (*this)->getAttr(getUpperBoundAttrStrName()).cast<AffineMapAttr>();
  }

  /// Set lower bound. The new bound must have the same number of operands as
  /// the current bound map. Otherwise, 'replaceForLowerBound' should be used.
  void setLowerBound(ValueRange operands, AffineMap map);
  /// Set upper bound. The new bound must not have more operands than the
  /// current bound map. Otherwise, 'replaceForUpperBound' should be used.
  void setUpperBound(ValueRange operands, AffineMap map);

  /// Set the lower bound map without changing operands.
  void setLowerBoundMap(AffineMap map);

  /// Set the upper bound map without changing operands.
  void setUpperBoundMap(AffineMap map);

  /// Set loop step.
  void setStep(int64_t step) {
    assert(step > 0 && "step has to be a positive integer constant");
    auto *context = getLowerBoundMap().getContext();
    (*this)->setAttr(StringAttr::get(context, getStepAttrStrName()),
                     IntegerAttr::get(IndexType::get(context), step));
  }

  /// Returns number of region arguments for loop-carried values.
  unsigned getNumRegionIterArgs() {
    return getBody()->getNumArguments() - 1;
  }

  /// Number of operands controlling the loop: lb and ub.
  unsigned getNumControlOperands() { return getOperation()->getNumOperands() - getNumIterOperands(); }

  /// Get the number of loop-carried values.
  unsigned getNumIterOperands();

  /// Returns true if the lower bound is constant.
  bool hasConstantLowerBound();
  /// Returns true if the upper bound is constant.
  bool hasConstantUpperBound();
  /// Returns true if both bounds are constant.
  bool hasConstantBounds() {
    return hasConstantLowerBound() && hasConstantUpperBound();
  }
  /// Returns the value of the constant lower bound.
  /// Fails assertion if the bound is non-constant.
  int64_t getConstantLowerBound();
  /// Returns the value of the constant upper bound. The upper bound is
  /// exclusive. Fails assertion if the bound is non-constant.
  int64_t getConstantUpperBound();
  /// Sets the lower bound to the given constant value.
  void setConstantLowerBound(int64_t value);
  /// Sets the upper bound to the given constant value.
  void setConstantUpperBound(int64_t value);

  /// Returns true if both the lower and upper bound have the same operand
  /// lists (same operands in the same order).
  bool matchingBoundOperandList();

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineForOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineIfOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineIfOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineIfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AffineIfOpGenericAdaptor : public detail::AffineIfOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineIfOpGenericAdaptorBase;
public:
  AffineIfOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineIfOpAdaptor : public AffineIfOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineIfOpGenericAdaptor::AffineIfOpGenericAdaptor;
  AffineIfOpAdaptor(AffineIfOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineIfOp : public ::mlir::Op<AffineIfOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::RecursivelySpeculatableImplTrait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineIfOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineIfOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.if");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, IntegerSet set, ValueRange args, bool withElseRegion);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, IntegerSet set, ValueRange args, bool withElseRegion);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  static StringRef getConditionAttrStrName() { return "condition"; }

  IntegerSet getIntegerSet();
  void setIntegerSet(IntegerSet newSet);

  /// Sets the integer set with its operands.
  void setConditional(IntegerSet set, ValueRange operands);

  /// Returns true if an else block exists.
  bool hasElse() { return !getElseRegion().empty(); }

  Block *getThenBlock() {
    assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
    return &getThenRegion().front();
  }

  Block *getElseBlock() {
    assert(hasElse() && "Empty 'else' region.");
    return &getElseRegion().front();
  }

  OpBuilder getThenBodyBuilder() {
    assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
    Block &body = getThenRegion().front();
    return OpBuilder(&body, std::prev(body.end()));
  }
  OpBuilder getElseBodyBuilder() {
    assert(hasElse() && "No 'else' block");
    Block &body = getElseRegion().front();
    return OpBuilder(&body, std::prev(body.end()));
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineIfOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineLoadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AffineLoadOpGenericAdaptor : public detail::AffineLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineLoadOpGenericAdaptorBase;
public:
  AffineLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineLoadOpAdaptor : public AffineLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineLoadOpGenericAdaptor::AffineLoadOpGenericAdaptor;
  AffineLoadOpAdaptor(AffineLoadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineLoadOp : public ::mlir::Op<AffineLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::affine::AffineReadOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap map, ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, ValueRange indices = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 0; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return (*this)->getAttr(getMapAttrStrName()).cast<AffineMapAttr>();
  }

  static StringRef getMapAttrStrName() { return "map"; }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMaxOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineMaxOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineMaxOpGenericAdaptor : public detail::AffineMaxOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineMaxOpGenericAdaptorBase;
public:
  AffineMaxOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class AffineMaxOpAdaptor : public AffineMaxOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineMaxOpGenericAdaptor::AffineMaxOpGenericAdaptor;
  AffineMaxOpAdaptor(AffineMaxOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineMaxOp : public ::mlir::Op<AffineMaxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineMaxOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineMaxOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.max");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getOperands();
  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr);
  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap affineMap, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getMapAttrStrName() { return "map"; }
  AffineMap getAffineMap() { return getMap(); }
  ValueRange getMapOperands() { return getOperands(); }
  ValueRange getDimOperands() {
    return OperandRange{getOperands().begin(),
                        getOperands().begin() + getMap().getNumDims()};
  }
  ValueRange getSymbolOperands() {
    return OperandRange{getOperands().begin() + getMap().getNumDims(),
                        getOperands().end()};
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMaxOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMinOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineMinOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
};
} // namespace detail
template <typename RangeT>
class AffineMinOpGenericAdaptor : public detail::AffineMinOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineMinOpGenericAdaptorBase;
public:
  AffineMinOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class AffineMinOpAdaptor : public AffineMinOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineMinOpGenericAdaptor::AffineMinOpGenericAdaptor;
  AffineMinOpAdaptor(AffineMinOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineMinOp : public ::mlir::Op<AffineMinOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineMinOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineMinOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMapAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.min");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getOperands();
  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr);
  void setMap(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap affineMap, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getMapAttrStrName() { return "map"; }
  AffineMap getAffineMap() { return getMap(); }
  ValueRange getMapOperands() { return getOperands(); }
  ValueRange getDimOperands() {
    return OperandRange{getOperands().begin(),
                        getOperands().begin() + getMap().getNumDims()};
  }
  ValueRange getSymbolOperands() {
    return OperandRange{getOperands().begin() + getMap().getNumDims(),
                        getOperands().end()};
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMinOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineParallelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineParallelOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReductionsAttr();
  ::mlir::ArrayAttr getReductions();
  ::mlir::AffineMapAttr getLowerBoundsMapAttr();
  ::mlir::AffineMap getLowerBoundsMap();
  ::mlir::DenseIntElementsAttr getLowerBoundsGroupsAttr();
  ::mlir::DenseIntElementsAttr getLowerBoundsGroups();
  ::mlir::AffineMapAttr getUpperBoundsMapAttr();
  ::mlir::AffineMap getUpperBoundsMap();
  ::mlir::DenseIntElementsAttr getUpperBoundsGroupsAttr();
  ::mlir::DenseIntElementsAttr getUpperBoundsGroups();
  ::mlir::ArrayAttr getStepsAttr();
  ::llvm::SmallVector<int64_t, 8> getSteps();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AffineParallelOpGenericAdaptor : public detail::AffineParallelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineParallelOpGenericAdaptorBase;
public:
  AffineParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getMapOperands() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineParallelOpAdaptor : public AffineParallelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineParallelOpGenericAdaptor::AffineParallelOpGenericAdaptor;
  AffineParallelOpAdaptor(AffineParallelOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineParallelOp : public ::mlir::Op<AffineParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::RecursivelySpeculatableImplTrait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lowerBoundsGroups"), ::llvm::StringRef("lowerBoundsMap"), ::llvm::StringRef("reductions"), ::llvm::StringRef("steps"), ::llvm::StringRef("upperBoundsGroups"), ::llvm::StringRef("upperBoundsMap")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getLowerBoundsGroupsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getLowerBoundsGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getLowerBoundsMapAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getLowerBoundsMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getReductionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getReductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStepsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStepsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getUpperBoundsGroupsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getUpperBoundsGroupsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getUpperBoundsMapAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getUpperBoundsMapAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getMapOperands();
  ::mlir::MutableOperandRange getMapOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  ::mlir::ArrayAttr getReductionsAttr();
  ::mlir::ArrayAttr getReductions();
  ::mlir::AffineMapAttr getLowerBoundsMapAttr();
  ::mlir::AffineMap getLowerBoundsMap();
  ::mlir::DenseIntElementsAttr getLowerBoundsGroupsAttr();
  ::mlir::DenseIntElementsAttr getLowerBoundsGroups();
  ::mlir::AffineMapAttr getUpperBoundsMapAttr();
  ::mlir::AffineMap getUpperBoundsMap();
  ::mlir::DenseIntElementsAttr getUpperBoundsGroupsAttr();
  ::mlir::DenseIntElementsAttr getUpperBoundsGroups();
  ::mlir::ArrayAttr getStepsAttr();
  ::llvm::SmallVector<int64_t, 8> getSteps();
  void setReductionsAttr(::mlir::ArrayAttr attr);
  void setLowerBoundsMapAttr(::mlir::AffineMapAttr attr);
  void setLowerBoundsMap(::mlir::AffineMap attrValue);
  void setLowerBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setUpperBoundsMapAttr(::mlir::AffineMapAttr attr);
  void setUpperBoundsMap(::mlir::AffineMap attrValue);
  void setUpperBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setStepsAttr(::mlir::ArrayAttr attr);
  void setSteps(::llvm::SmallVector<int64_t, 8> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ArrayRef<arith::AtomicRMWKind> reductions, ArrayRef<int64_t> ranges);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ArrayRef<arith::AtomicRMWKind> reductions, ArrayRef<AffineMap> lbMaps, ValueRange lbArgs, ArrayRef<AffineMap> ubMaps, ValueRange ubArgs, ArrayRef<int64_t> steps);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMapAttr lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMapAttr upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::mlir::ArrayAttr steps, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMap lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMap upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::llvm::SmallVector<int64_t, 8> steps, ::mlir::ValueRange mapOperands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::mlir::Region &getLoopBody();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Get the number of dimensions.
  unsigned getNumDims();

  /// Get ranges as constants, may fail in dynamic case.
  std::optional<SmallVector<int64_t, 8>> getConstantRanges();

  Block *getBody();
  OpBuilder getBodyBuilder();
  MutableArrayRef<BlockArgument> getIVs() {
    return getBody()->getArguments();
  }

  /// Returns elements of the loop lower bound.
  AffineMap getLowerBoundMap(unsigned pos);
  operand_range getLowerBoundsOperands();
  AffineValueMap getLowerBoundsValueMap();

  /// Sets elements of the loop lower bound.
  void setLowerBounds(ValueRange operands, AffineMap map);

  /// Returns elements of the loop upper bound.
  AffineMap getUpperBoundMap(unsigned pos);
  operand_range getUpperBoundsOperands();
  AffineValueMap getUpperBoundsValueMap();

  /// Sets elements fo the loop upper bound.
  void setUpperBounds(ValueRange operands, AffineMap map);

  void setSteps(ArrayRef<int64_t> newSteps);

  /// Returns attribute names to use in op construction. Not expected to be
  /// used directly.
  static StringRef getReductionsAttrStrName() { return "reductions"; }
  static StringRef getLowerBoundsMapAttrStrName() { return "lowerBoundsMap"; }
  static StringRef getLowerBoundsGroupsAttrStrName() {
    return "lowerBoundsGroups";
  }
  static StringRef getUpperBoundsMapAttrStrName() { return "upperBoundsMap"; }
  static StringRef getUpperBoundsGroupsAttrStrName() {
    return "upperBoundsGroups";
  }
  static StringRef getStepsAttrStrName() { return "steps"; }

  /// Returns `true` if the loop bounds have min/max expressions.
  bool hasMinMaxBounds() {
    return getLowerBoundsMap().getNumResults() != getNumDims() ||
           getUpperBoundsMap().getNumResults() != getNumDims();
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineParallelOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffinePrefetchOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffinePrefetchOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffinePrefetchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getIsWriteAttr();
  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr();
  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr();
  bool getIsDataCache();
};
} // namespace detail
template <typename RangeT>
class AffinePrefetchOpGenericAdaptor : public detail::AffinePrefetchOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffinePrefetchOpGenericAdaptorBase;
public:
  AffinePrefetchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffinePrefetchOpAdaptor : public AffinePrefetchOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffinePrefetchOpGenericAdaptor::AffinePrefetchOpGenericAdaptor;
  AffinePrefetchOpAdaptor(AffinePrefetchOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffinePrefetchOp : public ::mlir::Op<AffinePrefetchOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::affine::AffineMapAccessInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffinePrefetchOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffinePrefetchOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isDataCache"), ::llvm::StringRef("isWrite"), ::llvm::StringRef("localityHint")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsDataCacheAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsDataCacheAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsWriteAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsWriteAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLocalityHintAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLocalityHintAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.prefetch");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getIsWriteAttr();
  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr();
  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr();
  bool getIsDataCache();
  void setIsWriteAttr(::mlir::BoolAttr attr);
  void setIsWrite(bool attrValue);
  void setLocalityHintAttr(::mlir::IntegerAttr attr);
  void setLocalityHint(uint32_t attrValue);
  void setIsDataCacheAttr(::mlir::BoolAttr attr);
  void setIsDataCache(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ArrayRef<Value> mapOperands, bool isWrite, unsigned localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  MemRefType getMemRefType() {
    return getMemref().getType().cast<MemRefType>();
  }

  /// Returns the affine map used to index the memref for this operation.
  AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
  AffineMapAttr getAffineMapAttr() {
    return (*this)->getAttr(getMapAttrStrName()).cast<AffineMapAttr>();
  }

  /// Impelements the AffineMapAccessInterface.
  /// Returns the AffineMapAttr associated with 'memref'.
  NamedAttribute getAffineMapAttrForMemRef(Value mref) {
    assert(mref == getMemref() &&
           "Expected mref argument to match memref operand");
    return {StringAttr::get(getContext(), getMapAttrStrName()),
      getAffineMapAttr()};
  }

  /// Get affine map operands.
  operand_range getMapOperands() {
    return {operand_begin() + 1, operand_end()};
  }

  static StringRef getMapAttrStrName() { return "map"; }
  static StringRef getLocalityHintAttrStrName() { return "localityHint"; }
  static StringRef getIsWriteAttrStrName() { return "isWrite"; }
  static StringRef getIsDataCacheAttrStrName() { return "isDataCache"; }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffinePrefetchOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineStoreOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AffineStoreOpGenericAdaptor : public detail::AffineStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineStoreOpGenericAdaptorBase;
public:
  AffineStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineStoreOpAdaptor : public AffineStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineStoreOpGenericAdaptor::AffineStoreOpGenericAdaptor;
  AffineStoreOpAdaptor(AffineStoreOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineStoreOp : public ::mlir::Op<AffineStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::affine::AffineWriteOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, AffineMap map, ValueRange mapOperands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Returns the operand index of the value to be stored.
  unsigned getStoredValOperandIndex() { return 0; }

  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 1; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return (*this)->getAttr(getMapAttrStrName()).cast<AffineMapAttr>();
  }

  static StringRef getMapAttrStrName() { return "map"; }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorLoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineVectorLoadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineVectorLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AffineVectorLoadOpGenericAdaptor : public detail::AffineVectorLoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineVectorLoadOpGenericAdaptorBase;
public:
  AffineVectorLoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineVectorLoadOpAdaptor : public AffineVectorLoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineVectorLoadOpGenericAdaptor::AffineVectorLoadOpGenericAdaptor;
  AffineVectorLoadOpAdaptor(AffineVectorLoadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineVectorLoadOp : public ::mlir::Op<AffineVectorLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::affine::AffineReadOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineVectorLoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineVectorLoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.vector_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::VectorType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType resultType, AffineMap map, ValueRange operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType resultType, Value memref, ValueRange indices = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, VectorType resultType, Value memref, AffineMap map, ValueRange mapOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 0; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return (*this)->getAttr(getMapAttrStrName()).cast<AffineMapAttr>();
  }

  static StringRef getMapAttrStrName() { return "map"; }

  VectorType getVectorType() {
    return getResult().getType().cast<VectorType>();
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineVectorStoreOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineVectorStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AffineVectorStoreOpGenericAdaptor : public detail::AffineVectorStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineVectorStoreOpGenericAdaptorBase;
public:
  AffineVectorStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AffineVectorStoreOpAdaptor : public AffineVectorStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineVectorStoreOpGenericAdaptor::AffineVectorStoreOpGenericAdaptor;
  AffineVectorStoreOpAdaptor(AffineVectorStoreOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineVectorStoreOp : public ::mlir::Op<AffineVectorStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::affine::AffineWriteOpInterface::Trait, ::mlir::affine::AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineVectorStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineVectorStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.vector_store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::VectorType> getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref, AffineMap map, ValueRange mapOperands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Returns the operand index of the value to be stored.
  unsigned getStoredValOperandIndex() { return 0; }

  /// Returns the operand index of the memref.
  unsigned getMemRefOperandIndex() { return 1; }

  void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

  /// Returns the affine map used to index the memref for this operation.
  AffineMapAttr getAffineMapAttr() {
    return (*this)->getAttr(getMapAttrStrName()).cast<AffineMapAttr>();
  }

  static StringRef getMapAttrStrName() { return "map"; }

  VectorType getVectorType() {
    return getValue().getType().cast<VectorType>();
  }
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineYieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AffineYieldOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AffineYieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AffineYieldOpGenericAdaptor : public detail::AffineYieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AffineYieldOpGenericAdaptorBase;
public:
  AffineYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class AffineYieldOpAdaptor : public AffineYieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AffineYieldOpGenericAdaptor::AffineYieldOpGenericAdaptor;
  AffineYieldOpAdaptor(AffineYieldOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AffineYieldOp : public ::mlir::Op<AffineYieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineYieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AffineYieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getOperands();
  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace affine
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::affine::AffineYieldOp)


#endif  // GET_OP_CLASSES

