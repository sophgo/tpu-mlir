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
class AffineApplyOp;
} // namespace mlir
namespace mlir {
class AffineDelinearizeIndexOp;
} // namespace mlir
namespace mlir {
class AffineForOp;
} // namespace mlir
namespace mlir {
class AffineIfOp;
} // namespace mlir
namespace mlir {
class AffineLoadOp;
} // namespace mlir
namespace mlir {
class AffineMaxOp;
} // namespace mlir
namespace mlir {
class AffineMinOp;
} // namespace mlir
namespace mlir {
class AffineParallelOp;
} // namespace mlir
namespace mlir {
class AffinePrefetchOp;
} // namespace mlir
namespace mlir {
class AffineStoreOp;
} // namespace mlir
namespace mlir {
class AffineVectorLoadOp;
} // namespace mlir
namespace mlir {
class AffineVectorStoreOp;
} // namespace mlir
namespace mlir {
class AffineYieldOp;
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineApplyOp declarations
//===----------------------------------------------------------------------===//

class AffineApplyOpAdaptor {
public:
  AffineApplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineApplyOpAdaptor(AffineApplyOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getMapOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineApplyOp : public ::mlir::Op<AffineApplyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineApplyOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::makeArrayRef(attrNames);
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
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
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
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineApplyOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineDelinearizeIndexOp declarations
//===----------------------------------------------------------------------===//

class AffineDelinearizeIndexOpAdaptor {
public:
  AffineDelinearizeIndexOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineDelinearizeIndexOpAdaptor(AffineDelinearizeIndexOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getLinearIndex();
  ::mlir::ValueRange getBasis();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineDelinearizeIndexOp : public ::mlir::Op<AffineDelinearizeIndexOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineDelinearizeIndexOpAdaptor;
public:
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineDelinearizeIndexOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineForOp declarations
//===----------------------------------------------------------------------===//

class AffineForOpAdaptor {
public:
  AffineForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineForOpAdaptor(AffineForOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineForOp : public ::mlir::Op<AffineForOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::OpTrait::HasRecursiveSideEffects, ::mlir::LoopLikeOpInterface::Trait, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineForOpAdaptor;
public:
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, int64_t lowerBound, int64_t upperBound, int64_t step = 1, ValueRange iterArgs = llvm::None, function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lbOperands, AffineMap lbMap, ValueRange ubOperands, AffineMap ubMap, int64_t step = 1, ValueRange iterArgs = llvm::None, function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::mlir::Region &getLoopBody();
  ::llvm::Optional<::mlir::Value> getSingleInductionVar();
  ::llvm::Optional<::mlir::OpFoldResult> getSingleLowerBound();
  ::llvm::Optional<::mlir::OpFoldResult> getSingleStep();
  ::llvm::Optional<::mlir::OpFoldResult> getSingleUpperBound();
  ::mlir::OperandRange getSuccessorEntryOperands(::llvm::Optional<unsigned> index);
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
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
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineForOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineIfOp declarations
//===----------------------------------------------------------------------===//

class AffineIfOpAdaptor {
public:
  AffineIfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineIfOpAdaptor(AffineIfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineIfOp : public ::mlir::Op<AffineIfOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineIfOpAdaptor;
public:
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
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineIfOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineLoadOp declarations
//===----------------------------------------------------------------------===//

class AffineLoadOpAdaptor {
public:
  AffineLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineLoadOpAdaptor(AffineLoadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineLoadOp : public ::mlir::Op<AffineLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, AffineReadOpInterface::Trait, AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineLoadOpAdaptor;
public:
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
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineLoadOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineMaxOp declarations
//===----------------------------------------------------------------------===//

class AffineMaxOpAdaptor {
public:
  AffineMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineMaxOpAdaptor(AffineMaxOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineMaxOp : public ::mlir::Op<AffineMaxOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineMaxOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::makeArrayRef(attrNames);
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
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr);
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
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  static StringRef getMapAttrStrName() { return "map"; }
  AffineMap getAffineMap() { return getMap(); }
  ValueRange getMapOperands() { return operands(); }
  ValueRange getDimOperands() {
    return OperandRange{operands().begin(),
                      operands().begin() + getMap().getNumDims()};
  }
  ValueRange getSymbolOperands() {
    return OperandRange{operands().begin() + getMap().getNumDims(),
                      operands().end()};
  }
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineMaxOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineMinOp declarations
//===----------------------------------------------------------------------===//

class AffineMinOpAdaptor {
public:
  AffineMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineMinOpAdaptor(AffineMinOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineMinOp : public ::mlir::Op<AffineMinOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineMinOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("map")};
    return ::llvm::makeArrayRef(attrNames);
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
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr getMapAttr();
  ::mlir::AffineMap getMap();
  void setMapAttr(::mlir::AffineMapAttr attr);
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
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  static StringRef getMapAttrStrName() { return "map"; }
  AffineMap getAffineMap() { return getMap(); }
  ValueRange getMapOperands() { return operands(); }
  ValueRange getDimOperands() {
    return OperandRange{operands().begin(),
                      operands().begin() + getMap().getNumDims()};
  }
  ValueRange getSymbolOperands() {
    return OperandRange{operands().begin() + getMap().getNumDims(),
                      operands().end()};
  }
};
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineMinOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineParallelOp declarations
//===----------------------------------------------------------------------===//

class AffineParallelOpAdaptor {
public:
  AffineParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineParallelOpAdaptor(AffineParallelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getMapOperands();
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
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineParallelOp : public ::mlir::Op<AffineParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AffineYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::OpTrait::HasRecursiveSideEffects, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("lowerBoundsGroups"), ::llvm::StringRef("lowerBoundsMap"), ::llvm::StringRef("reductions"), ::llvm::StringRef("steps"), ::llvm::StringRef("upperBoundsGroups"), ::llvm::StringRef("upperBoundsMap")};
    return ::llvm::makeArrayRef(attrNames);
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
  void setLowerBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setUpperBoundsMapAttr(::mlir::AffineMapAttr attr);
  void setUpperBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr);
  void setStepsAttr(::mlir::ArrayAttr attr);
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
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  ::mlir::Region &getLoopBody();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// Get the number of dimensions.
  unsigned getNumDims();

  /// Get ranges as constants, may fail in dynamic case.
  Optional<SmallVector<int64_t, 8>> getConstantRanges();

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
  void setLowerBoundsMap(AffineMap map);

  /// Returns elements of the loop upper bound.
  AffineMap getUpperBoundMap(unsigned pos);
  operand_range getUpperBoundsOperands();
  AffineValueMap getUpperBoundsValueMap();

  /// Sets elements fo the loop upper bound.
  void setUpperBounds(ValueRange operands, AffineMap map);
  void setUpperBoundsMap(AffineMap map);

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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineParallelOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffinePrefetchOp declarations
//===----------------------------------------------------------------------===//

class AffinePrefetchOpAdaptor {
public:
  AffinePrefetchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffinePrefetchOpAdaptor(AffinePrefetchOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getIsWriteAttr();
  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr();
  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr();
  bool getIsDataCache();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffinePrefetchOp : public ::mlir::Op<AffinePrefetchOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, AffineMapAccessInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffinePrefetchOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isDataCache"), ::llvm::StringRef("isWrite"), ::llvm::StringRef("localityHint")};
    return ::llvm::makeArrayRef(attrNames);
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
  void setLocalityHintAttr(::mlir::IntegerAttr attr);
  void setIsDataCacheAttr(::mlir::BoolAttr attr);
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
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffinePrefetchOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineStoreOp declarations
//===----------------------------------------------------------------------===//

class AffineStoreOpAdaptor {
public:
  AffineStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineStoreOpAdaptor(AffineStoreOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineStoreOp : public ::mlir::Op<AffineStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, AffineWriteOpInterface::Trait, AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineStoreOpAdaptor;
public:
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
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineStoreOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineVectorLoadOp declarations
//===----------------------------------------------------------------------===//

class AffineVectorLoadOpAdaptor {
public:
  AffineVectorLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineVectorLoadOpAdaptor(AffineVectorLoadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineVectorLoadOp : public ::mlir::Op<AffineVectorLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::VectorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, AffineReadOpInterface::Trait, AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineVectorLoadOpAdaptor;
public:
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineVectorLoadOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineVectorStoreOp declarations
//===----------------------------------------------------------------------===//

class AffineVectorStoreOpAdaptor {
public:
  AffineVectorStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineVectorStoreOpAdaptor(AffineVectorStoreOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineVectorStoreOp : public ::mlir::Op<AffineVectorStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, AffineWriteOpInterface::Trait, AffineMapAccessInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineVectorStoreOpAdaptor;
public:
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineVectorStoreOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineYieldOp declarations
//===----------------------------------------------------------------------===//

class AffineYieldOpAdaptor {
public:
  AffineYieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AffineYieldOpAdaptor(AffineYieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AffineYieldOp : public ::mlir::Op<AffineYieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::MemRefsNormalizable> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AffineYieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("affine.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
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
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(mlir::AffineYieldOp)


#endif  // GET_OP_CLASSES

