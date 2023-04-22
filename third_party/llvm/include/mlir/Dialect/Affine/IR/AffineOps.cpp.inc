/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::affine::AffineApplyOp,
::mlir::affine::AffineDelinearizeIndexOp,
::mlir::affine::AffineForOp,
::mlir::affine::AffineIfOp,
::mlir::affine::AffineLoadOp,
::mlir::affine::AffineMaxOp,
::mlir::affine::AffineMinOp,
::mlir::affine::AffineParallelOp,
::mlir::affine::AffinePrefetchOp,
::mlir::affine::AffineStoreOp,
::mlir::affine::AffineVectorLoadOp,
::mlir::affine::AffineVectorStoreOp,
::mlir::affine::AffineYieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace affine {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AffineOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AffineOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AffineOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AffineOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AffineOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::AffineMapAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AffineOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::arith::AtomicRMWKindAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Reduction ops";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AffineOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AffineOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AffineOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AffineOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((attr.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((attr.cast<::mlir::IntegerAttr>().getInt() <= 3)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_AffineOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_AffineOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace affine
} // namespace mlir
namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineApplyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineApplyOpGenericAdaptorBase::AffineApplyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.apply", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineApplyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineApplyOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr AffineApplyOpGenericAdaptorBase::getMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AffineApplyOp::getMapAttrName(*odsOpName)).cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineApplyOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineApplyOpAdaptor::AffineApplyOpAdaptor(AffineApplyOp op) : AffineApplyOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineApplyOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.apply' op ""requires attribute 'map'");
    if (namedAttrIt->getName() == AffineApplyOp::getMapAttrName(*odsOpName)) {
      tblgen_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_map && !((tblgen_map.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'affine.apply' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineApplyOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineApplyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AffineApplyOp::getMapOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineApplyOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineApplyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineApplyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::AffineMapAttr AffineApplyOp::getMapAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineApplyOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineApplyOp::setMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getMapAttrName(), attr);
}

void AffineApplyOp::setMap(::mlir::AffineMap attrValue) {
  (*this)->setAttr(getMapAttrName(), ::mlir::AffineMapAttr::get(attrValue));
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap map, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(), map, mapOperands);
    
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<AffineExpr>  exprList, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(),
            AffineMap::inferFromExprList(exprList).front(), mapOperands);
    
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(getMapAttrName(odsState.name), map);
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(getMapAttrName(odsState.name), map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(getMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(getMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineApplyOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'map'");
    if (namedAttrIt->getName() == getMapAttrName()) {
      tblgen_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineApplyOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineApplyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineApplyOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineDelinearizeIndexOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineDelinearizeIndexOpGenericAdaptorBase::AffineDelinearizeIndexOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.delinearize_index", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineDelinearizeIndexOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineDelinearizeIndexOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AffineDelinearizeIndexOpAdaptor::AffineDelinearizeIndexOpAdaptor(AffineDelinearizeIndexOp op) : AffineDelinearizeIndexOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineDelinearizeIndexOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineDelinearizeIndexOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineDelinearizeIndexOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> AffineDelinearizeIndexOp::getLinearIndex() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range AffineDelinearizeIndexOp::getBasis() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffineDelinearizeIndexOp::getLinearIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineDelinearizeIndexOp::getBasisMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineDelinearizeIndexOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range AffineDelinearizeIndexOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range AffineDelinearizeIndexOp::getMultiIndex() {
  return getODSResults(0);
}

void AffineDelinearizeIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange multi_index, ::mlir::Value linear_index, ::mlir::ValueRange basis) {
  odsState.addOperands(linear_index);
  odsState.addOperands(basis);
  odsState.addTypes(multi_index);
}

void AffineDelinearizeIndexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineDelinearizeIndexOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineDelinearizeIndexOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AffineDelinearizeIndexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand linear_indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> linear_indexOperands(linear_indexRawOperands);  ::llvm::SMLoc linear_indexOperandsLoc;
  (void)linear_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> basisOperands;
  ::llvm::SMLoc basisOperandsLoc;
  (void)basisOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> multi_indexTypes;

  linear_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(linear_indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  basisOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(basisOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(multi_indexTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(multi_indexTypes);
  if (parser.resolveOperands(linear_indexOperands, odsBuildableType0, linear_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(basisOperands, odsBuildableType0, basisOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineDelinearizeIndexOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLinearIndex();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << "(";
  _odsPrinter << getBasis();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getMultiIndex().getTypes();
}

void AffineDelinearizeIndexOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineDelinearizeIndexOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineForOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineForOpGenericAdaptorBase::AffineForOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.for", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineForOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineForOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &AffineForOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange AffineForOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
AffineForOpAdaptor::AffineForOpAdaptor(AffineForOp op) : AffineForOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineForOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineForOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineForOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AffineForOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range AffineForOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range AffineForOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &AffineForOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::LogicalResult AffineForOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineForOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineForOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineIfOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineIfOpGenericAdaptorBase::AffineIfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.if", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineIfOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineIfOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &AffineIfOpGenericAdaptorBase::getThenRegion() {
  return *odsRegions[0];
}

::mlir::Region &AffineIfOpGenericAdaptorBase::getElseRegion() {
  return *odsRegions[1];
}

::mlir::RegionRange AffineIfOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
AffineIfOpAdaptor::AffineIfOpAdaptor(AffineIfOp op) : AffineIfOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineIfOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineIfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineIfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AffineIfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range AffineIfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range AffineIfOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &AffineIfOp::getThenRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &AffineIfOp::getElseRegion() {
  return (*this)->getRegion(1);
}

::mlir::LogicalResult AffineIfOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps0(*this, region, "thenRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps1(*this, region, "elseRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineIfOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineIfOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineLoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineLoadOpGenericAdaptorBase::AffineLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.load", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineLoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineLoadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AffineLoadOpAdaptor::AffineLoadOpAdaptor(AffineLoadOp op) : AffineLoadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> AffineLoadOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range AffineLoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffineLoadOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffineLoadOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void AffineLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void AffineLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMaxOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineMaxOpGenericAdaptorBase::AffineMaxOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.max", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineMaxOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineMaxOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr AffineMaxOpGenericAdaptorBase::getMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AffineMaxOp::getMapAttrName(*odsOpName)).cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineMaxOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineMaxOpAdaptor::AffineMaxOpAdaptor(AffineMaxOp op) : AffineMaxOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineMaxOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.max' op ""requires attribute 'map'");
    if (namedAttrIt->getName() == AffineMaxOp::getMapAttrName(*odsOpName)) {
      tblgen_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_map && !((tblgen_map.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'affine.max' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineMaxOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AffineMaxOp::getOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineMaxOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::AffineMapAttr AffineMaxOp::getMapAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineMaxOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineMaxOp::setMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getMapAttrName(), attr);
}

void AffineMaxOp::setMap(::mlir::AffineMap attrValue) {
  (*this)->setAttr(getMapAttrName(), ::mlir::AffineMapAttr::get(attrValue));
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap affineMap, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(), affineMap, mapOperands);
    
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), map);
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineMaxOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'map'");
    if (namedAttrIt->getName() == getMapAttrName()) {
      tblgen_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineMaxOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMaxOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineMinOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineMinOpGenericAdaptorBase::AffineMinOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.min", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineMinOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineMinOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr AffineMinOpGenericAdaptorBase::getMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AffineMinOp::getMapAttrName(*odsOpName)).cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineMinOpGenericAdaptorBase::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

} // namespace detail
AffineMinOpAdaptor::AffineMinOpAdaptor(AffineMinOp op) : AffineMinOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineMinOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.min' op ""requires attribute 'map'");
    if (namedAttrIt->getName() == AffineMinOp::getMapAttrName(*odsOpName)) {
      tblgen_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_map && !((tblgen_map.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'affine.min' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineMinOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AffineMinOp::getOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineMinOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::AffineMapAttr AffineMinOp::getMapAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineMinOp::getMap() {
  auto attr = getMapAttr();
  return attr.getValue();
}

void AffineMinOp::setMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getMapAttrName(), attr);
}

void AffineMinOp::setMap(::mlir::AffineMap attrValue) {
  (*this)->setAttr(getMapAttrName(), ::mlir::AffineMapAttr::get(attrValue));
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap affineMap, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(), affineMap, mapOperands);
    
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), map);
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineMinOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_map;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'map'");
    if (namedAttrIt->getName() == getMapAttrName()) {
      tblgen_map = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_map, "map")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineMinOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineMinOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineParallelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineParallelOpGenericAdaptorBase::AffineParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.parallel", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineParallelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineParallelOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr AffineParallelOpGenericAdaptorBase::getReductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 3, AffineParallelOp::getReductionsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AffineParallelOpGenericAdaptorBase::getReductions() {
  auto attr = getReductionsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOpGenericAdaptorBase::getLowerBoundsMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 4, AffineParallelOp::getLowerBoundsMapAttrName(*odsOpName)).cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineParallelOpGenericAdaptorBase::getLowerBoundsMap() {
  auto attr = getLowerBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOpGenericAdaptorBase::getLowerBoundsGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 5, AffineParallelOp::getLowerBoundsGroupsAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr AffineParallelOpGenericAdaptorBase::getLowerBoundsGroups() {
  auto attr = getLowerBoundsGroupsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOpGenericAdaptorBase::getUpperBoundsMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 5, odsAttrs.end() - 0, AffineParallelOp::getUpperBoundsMapAttrName(*odsOpName)).cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineParallelOpGenericAdaptorBase::getUpperBoundsMap() {
  auto attr = getUpperBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOpGenericAdaptorBase::getUpperBoundsGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 1, AffineParallelOp::getUpperBoundsGroupsAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr AffineParallelOpGenericAdaptorBase::getUpperBoundsGroups() {
  auto attr = getUpperBoundsGroupsAttr();
  return attr;
}

::mlir::ArrayAttr AffineParallelOpGenericAdaptorBase::getStepsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 2, AffineParallelOp::getStepsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::llvm::SmallVector<int64_t, 8> AffineParallelOpGenericAdaptorBase::getSteps() {
  auto attr = getStepsAttr();
  return llvm::to_vector<4>(
      llvm::map_range(attr.getAsRange<mlir::IntegerAttr>(),
      [](IntegerAttr attr) { return attr.getInt(); }));;
}

::mlir::Region &AffineParallelOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange AffineParallelOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
AffineParallelOpAdaptor::AffineParallelOpAdaptor(AffineParallelOp op) : AffineParallelOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineParallelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_lowerBoundsGroups;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.parallel' op ""requires attribute 'lowerBoundsGroups'");
    if (namedAttrIt->getName() == AffineParallelOp::getLowerBoundsGroupsAttrName(*odsOpName)) {
      tblgen_lowerBoundsGroups = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_lowerBoundsMap;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.parallel' op ""requires attribute 'lowerBoundsMap'");
    if (namedAttrIt->getName() == AffineParallelOp::getLowerBoundsMapAttrName(*odsOpName)) {
      tblgen_lowerBoundsMap = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.parallel' op ""requires attribute 'reductions'");
    if (namedAttrIt->getName() == AffineParallelOp::getReductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_steps;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.parallel' op ""requires attribute 'steps'");
    if (namedAttrIt->getName() == AffineParallelOp::getStepsAttrName(*odsOpName)) {
      tblgen_steps = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_upperBoundsGroups;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.parallel' op ""requires attribute 'upperBoundsGroups'");
    if (namedAttrIt->getName() == AffineParallelOp::getUpperBoundsGroupsAttrName(*odsOpName)) {
      tblgen_upperBoundsGroups = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_upperBoundsMap;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.parallel' op ""requires attribute 'upperBoundsMap'");
    if (namedAttrIt->getName() == AffineParallelOp::getUpperBoundsMapAttrName(*odsOpName)) {
      tblgen_upperBoundsMap = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::arith::AtomicRMWKindAttr>())); }))))
    return emitError(loc, "'affine.parallel' op ""attribute 'reductions' failed to satisfy constraint: Reduction ops");

  if (tblgen_lowerBoundsMap && !((tblgen_lowerBoundsMap.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'affine.parallel' op ""attribute 'lowerBoundsMap' failed to satisfy constraint: AffineMap attribute");

  if (tblgen_lowerBoundsGroups && !(((tblgen_lowerBoundsGroups.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_lowerBoundsGroups.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'affine.parallel' op ""attribute 'lowerBoundsGroups' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_upperBoundsMap && !((tblgen_upperBoundsMap.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'affine.parallel' op ""attribute 'upperBoundsMap' failed to satisfy constraint: AffineMap attribute");

  if (tblgen_upperBoundsGroups && !(((tblgen_upperBoundsGroups.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_upperBoundsGroups.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'affine.parallel' op ""attribute 'upperBoundsGroups' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_steps && !(((tblgen_steps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_steps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'affine.parallel' op ""attribute 'steps' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineParallelOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineParallelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AffineParallelOp::getMapOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineParallelOp::getMapOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineParallelOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range AffineParallelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range AffineParallelOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &AffineParallelOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr AffineParallelOp::getReductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 3, getReductionsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AffineParallelOp::getReductions() {
  auto attr = getReductionsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOp::getLowerBoundsMapAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 4, getLowerBoundsMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineParallelOp::getLowerBoundsMap() {
  auto attr = getLowerBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOp::getLowerBoundsGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 5, getLowerBoundsGroupsAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr AffineParallelOp::getLowerBoundsGroups() {
  auto attr = getLowerBoundsGroupsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOp::getUpperBoundsMapAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 5, (*this)->getAttrs().end() - 0, getUpperBoundsMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineParallelOp::getUpperBoundsMap() {
  auto attr = getUpperBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOp::getUpperBoundsGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 1, getUpperBoundsGroupsAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr AffineParallelOp::getUpperBoundsGroups() {
  auto attr = getUpperBoundsGroupsAttr();
  return attr;
}

::mlir::ArrayAttr AffineParallelOp::getStepsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 2, getStepsAttrName()).cast<::mlir::ArrayAttr>();
}

::llvm::SmallVector<int64_t, 8> AffineParallelOp::getSteps() {
  auto attr = getStepsAttr();
  return llvm::to_vector<4>(
      llvm::map_range(attr.getAsRange<mlir::IntegerAttr>(),
      [](IntegerAttr attr) { return attr.getInt(); }));;
}

void AffineParallelOp::setReductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReductionsAttrName(), attr);
}

void AffineParallelOp::setLowerBoundsMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getLowerBoundsMapAttrName(), attr);
}

void AffineParallelOp::setLowerBoundsMap(::mlir::AffineMap attrValue) {
  (*this)->setAttr(getLowerBoundsMapAttrName(), ::mlir::AffineMapAttr::get(attrValue));
}

void AffineParallelOp::setLowerBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getLowerBoundsGroupsAttrName(), attr);
}

void AffineParallelOp::setUpperBoundsMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getUpperBoundsMapAttrName(), attr);
}

void AffineParallelOp::setUpperBoundsMap(::mlir::AffineMap attrValue) {
  (*this)->setAttr(getUpperBoundsMapAttrName(), ::mlir::AffineMapAttr::get(attrValue));
}

void AffineParallelOp::setUpperBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getUpperBoundsGroupsAttrName(), attr);
}

void AffineParallelOp::setStepsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStepsAttrName(), attr);
}

void AffineParallelOp::setSteps(::llvm::SmallVector<int64_t, 8> attrValue) {
  (*this)->setAttr(getStepsAttrName(), ::mlir::Builder((*this)->getContext()).getI64ArrayAttr(attrValue));
}

void AffineParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMapAttr lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMapAttr upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::mlir::ArrayAttr steps, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  odsState.addAttribute(getLowerBoundsMapAttrName(odsState.name), lowerBoundsMap);
  odsState.addAttribute(getLowerBoundsGroupsAttrName(odsState.name), lowerBoundsGroups);
  odsState.addAttribute(getUpperBoundsMapAttrName(odsState.name), upperBoundsMap);
  odsState.addAttribute(getUpperBoundsGroupsAttrName(odsState.name), upperBoundsGroups);
  odsState.addAttribute(getStepsAttrName(odsState.name), steps);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AffineParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMap lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMap upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::llvm::SmallVector<int64_t, 8> steps, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(getReductionsAttrName(odsState.name), reductions);
  odsState.addAttribute(getLowerBoundsMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(lowerBoundsMap));
  odsState.addAttribute(getLowerBoundsGroupsAttrName(odsState.name), lowerBoundsGroups);
  odsState.addAttribute(getUpperBoundsMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(upperBoundsMap));
  odsState.addAttribute(getUpperBoundsGroupsAttrName(odsState.name), upperBoundsGroups);
  odsState.addAttribute(getStepsAttrName(odsState.name), odsBuilder.getI64ArrayAttr(steps));
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AffineParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineParallelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_lowerBoundsGroups;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'lowerBoundsGroups'");
    if (namedAttrIt->getName() == getLowerBoundsGroupsAttrName()) {
      tblgen_lowerBoundsGroups = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_lowerBoundsMap;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'lowerBoundsMap'");
    if (namedAttrIt->getName() == getLowerBoundsMapAttrName()) {
      tblgen_lowerBoundsMap = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'reductions'");
    if (namedAttrIt->getName() == getReductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_steps;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'steps'");
    if (namedAttrIt->getName() == getStepsAttrName()) {
      tblgen_steps = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_upperBoundsGroups;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'upperBoundsGroups'");
    if (namedAttrIt->getName() == getUpperBoundsGroupsAttrName()) {
      tblgen_upperBoundsGroups = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_upperBoundsMap;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'upperBoundsMap'");
    if (namedAttrIt->getName() == getUpperBoundsMapAttrName()) {
      tblgen_upperBoundsMap = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_lowerBoundsMap, "lowerBoundsMap")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps2(*this, tblgen_lowerBoundsGroups, "lowerBoundsGroups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_upperBoundsMap, "upperBoundsMap")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps2(*this, tblgen_upperBoundsGroups, "upperBoundsGroups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps3(*this, tblgen_steps, "steps")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineParallelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineParallelOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffinePrefetchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffinePrefetchOpGenericAdaptorBase::AffinePrefetchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.prefetch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffinePrefetchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffinePrefetchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr AffinePrefetchOpGenericAdaptorBase::getIsWriteAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, AffinePrefetchOp::getIsWriteAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool AffinePrefetchOpGenericAdaptorBase::getIsWrite() {
  auto attr = getIsWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr AffinePrefetchOpGenericAdaptorBase::getLocalityHintAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, AffinePrefetchOp::getLocalityHintAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t AffinePrefetchOpGenericAdaptorBase::getLocalityHint() {
  auto attr = getLocalityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr AffinePrefetchOpGenericAdaptorBase::getIsDataCacheAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AffinePrefetchOp::getIsDataCacheAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool AffinePrefetchOpGenericAdaptorBase::getIsDataCache() {
  auto attr = getIsDataCacheAttr();
  return attr.getValue();
}

} // namespace detail
AffinePrefetchOpAdaptor::AffinePrefetchOpAdaptor(AffinePrefetchOp op) : AffinePrefetchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffinePrefetchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_isDataCache;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.prefetch' op ""requires attribute 'isDataCache'");
    if (namedAttrIt->getName() == AffinePrefetchOp::getIsDataCacheAttrName(*odsOpName)) {
      tblgen_isDataCache = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isWrite;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.prefetch' op ""requires attribute 'isWrite'");
    if (namedAttrIt->getName() == AffinePrefetchOp::getIsWriteAttrName(*odsOpName)) {
      tblgen_isWrite = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_localityHint;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'affine.prefetch' op ""requires attribute 'localityHint'");
    if (namedAttrIt->getName() == AffinePrefetchOp::getLocalityHintAttrName(*odsOpName)) {
      tblgen_localityHint = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_isWrite && !((tblgen_isWrite.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'affine.prefetch' op ""attribute 'isWrite' failed to satisfy constraint: bool attribute");

  if (tblgen_localityHint && !((((tblgen_localityHint.isa<::mlir::IntegerAttr>())) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() <= 3))))
    return emitError(loc, "'affine.prefetch' op ""attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");

  if (tblgen_isDataCache && !((tblgen_isDataCache.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'affine.prefetch' op ""attribute 'isDataCache' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffinePrefetchOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffinePrefetchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> AffinePrefetchOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range AffinePrefetchOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffinePrefetchOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffinePrefetchOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffinePrefetchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffinePrefetchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr AffinePrefetchOp::getIsWriteAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIsWriteAttrName()).cast<::mlir::BoolAttr>();
}

bool AffinePrefetchOp::getIsWrite() {
  auto attr = getIsWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr AffinePrefetchOp::getLocalityHintAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getLocalityHintAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t AffinePrefetchOp::getLocalityHint() {
  auto attr = getLocalityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr AffinePrefetchOp::getIsDataCacheAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getIsDataCacheAttrName()).cast<::mlir::BoolAttr>();
}

bool AffinePrefetchOp::getIsDataCache() {
  auto attr = getIsDataCacheAttr();
  return attr.getValue();
}

void AffinePrefetchOp::setIsWriteAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getIsWriteAttrName(), attr);
}

void AffinePrefetchOp::setIsWrite(bool attrValue) {
  (*this)->setAttr(getIsWriteAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void AffinePrefetchOp::setLocalityHintAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLocalityHintAttrName(), attr);
}

void AffinePrefetchOp::setLocalityHint(uint32_t attrValue) {
  (*this)->setAttr(getLocalityHintAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AffinePrefetchOp::setIsDataCacheAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getIsDataCacheAttrName(), attr);
}

void AffinePrefetchOp::setIsDataCache(bool attrValue) {
  (*this)->setAttr(getIsDataCacheAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ArrayRef<Value> mapOperands, bool isWrite, unsigned localityHint, bool isDataCache) {
      assert(map.getNumInputs() == mapOperands.size()
             && "inconsistent index info");
      auto localityHintAttr = odsBuilder.getI32IntegerAttr(localityHint);
      auto isWriteAttr = odsBuilder.getBoolAttr(isWrite);
      auto isDataCacheAttr = odsBuilder.getBoolAttr(isDataCache);
      odsState.addOperands(memref);
      odsState.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
      odsState.addOperands(mapOperands);
      odsState.addAttribute(getLocalityHintAttrStrName(), localityHintAttr);
      odsState.addAttribute(getIsWriteAttrStrName(), isWriteAttr);
      odsState.addAttribute(getIsDataCacheAttrStrName(), isDataCacheAttr);
    
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), isWrite);
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), localityHint);
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), isDataCache);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), isWrite);
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), localityHint);
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), isDataCache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), odsBuilder.getBoolAttr(isDataCache));
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), odsBuilder.getBoolAttr(isDataCache));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffinePrefetchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_isDataCache;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'isDataCache'");
    if (namedAttrIt->getName() == getIsDataCacheAttrName()) {
      tblgen_isDataCache = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isWrite;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'isWrite'");
    if (namedAttrIt->getName() == getIsWriteAttrName()) {
      tblgen_isWrite = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_localityHint;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'localityHint'");
    if (namedAttrIt->getName() == getLocalityHintAttrName()) {
      tblgen_localityHint = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(*this, tblgen_isWrite, "isWrite")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps5(*this, tblgen_localityHint, "localityHint")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(*this, tblgen_isDataCache, "isDataCache")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffinePrefetchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffinePrefetchOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineStoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineStoreOpGenericAdaptorBase::AffineStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.store", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineStoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineStoreOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AffineStoreOpAdaptor::AffineStoreOpAdaptor(AffineStoreOp op) : AffineStoreOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffineStoreOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> AffineStoreOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range AffineStoreOp::getIndices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AffineStoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineStoreOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult AffineStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorLoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineVectorLoadOpGenericAdaptorBase::AffineVectorLoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.vector_load", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineVectorLoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineVectorLoadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AffineVectorLoadOpAdaptor::AffineVectorLoadOpAdaptor(AffineVectorLoadOp op) : AffineVectorLoadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineVectorLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineVectorLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineVectorLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> AffineVectorLoadOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range AffineVectorLoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffineVectorLoadOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineVectorLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineVectorLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineVectorLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::VectorType> AffineVectorLoadOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::VectorType>>(*getODSResults(0).begin());
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineVectorLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineVectorLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineVectorLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineVectorLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorLoadOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineVectorStoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineVectorStoreOpGenericAdaptorBase::AffineVectorStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.vector_store", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineVectorStoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineVectorStoreOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AffineVectorStoreOpAdaptor::AffineVectorStoreOpAdaptor(AffineVectorStoreOp op) : AffineVectorStoreOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineVectorStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineVectorStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineVectorStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::VectorType> AffineVectorStoreOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::VectorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> AffineVectorStoreOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range AffineVectorStoreOp::getIndices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AffineVectorStoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineVectorStoreOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineVectorStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineVectorStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineVectorStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult AffineVectorStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineVectorStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AffineVectorStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineVectorStoreOp)

namespace mlir {
namespace affine {

//===----------------------------------------------------------------------===//
// ::mlir::affine::AffineYieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AffineYieldOpGenericAdaptorBase::AffineYieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("affine.yield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AffineYieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AffineYieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AffineYieldOpAdaptor::AffineYieldOpAdaptor(AffineYieldOp op) : AffineYieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AffineYieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AffineYieldOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AffineYieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AffineYieldOp::getOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineYieldOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AffineYieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AffineYieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AffineYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 build(odsBuilder, odsState, std::nullopt); 
}

void AffineYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void AffineYieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineYieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AffineYieldOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AffineYieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AffineYieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getOperands().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getOperands().getTypes();
  }
}

void AffineYieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace affine
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::affine::AffineYieldOp)


#endif  // GET_OP_CLASSES

