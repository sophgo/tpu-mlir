/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

mlir::AffineApplyOp,
mlir::AffineForOp,
mlir::AffineIfOp,
mlir::AffineLoadOp,
mlir::AffineMaxOp,
mlir::AffineMinOp,
mlir::AffineParallelOp,
mlir::AffinePrefetchOp,
mlir::AffineStoreOp,
mlir::AffineVectorLoadOp,
mlir::AffineVectorStoreOp,
mlir::AffineYieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {

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
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::arith::AtomicRMWKindAttr>()); })))) {
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
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) {
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
} // namespace mlir
namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineApplyOp definitions
//===----------------------------------------------------------------------===//

AffineApplyOpAdaptor::AffineApplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineApplyOpAdaptor::AffineApplyOpAdaptor(AffineApplyOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineApplyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineApplyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineApplyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AffineApplyOpAdaptor::mapOperands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr AffineApplyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr AffineApplyOpAdaptor::mapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineApplyOpAdaptor::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

::mlir::LogicalResult AffineApplyOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_map = odsAttrs.get("map");
    if (!tblgen_map)
      return emitError(loc, "'affine.apply' op ""requires attribute 'map'");

    if (tblgen_map && !((tblgen_map.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'affine.apply' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  }
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

::mlir::Operation::operand_range AffineApplyOp::mapOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineApplyOp::mapOperandsMutable() {
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

::mlir::AffineMapAttr AffineApplyOp::mapAttr() {
  return (*this)->getAttr(mapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineApplyOp::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

void AffineApplyOp::mapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(mapAttrName(), attr);
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
  odsState.addAttribute(mapAttrName(odsState.name), map);
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(mapAttrName(odsState.name), map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(mapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(mapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineApplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineApplyOp::verifyInvariants() {
  {
    auto tblgen_map = (*this)->getAttr(mapAttrName());
    if (!tblgen_map)
      return emitOpError("requires attribute 'map'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_map, "map")))
      return ::mlir::failure();
  }
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
  return verify();
}

void AffineApplyOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineApplyOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineForOp definitions
//===----------------------------------------------------------------------===//

AffineForOpAdaptor::AffineForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineForOpAdaptor::AffineForOpAdaptor(AffineForOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineForOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineForOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineForOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AffineForOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange AffineForOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &AffineForOpAdaptor::region() {
  return *odsRegions[0];
}

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

::mlir::Operation::result_range AffineForOp::results() {
  return getODSResults(0);
}

::mlir::Region &AffineForOp::region() {
  return (*this)->getRegion(0);
}

::mlir::LogicalResult AffineForOp::verifyInvariants() {
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return verify();
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineForOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineIfOp definitions
//===----------------------------------------------------------------------===//

AffineIfOpAdaptor::AffineIfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineIfOpAdaptor::AffineIfOpAdaptor(AffineIfOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineIfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineIfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineIfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AffineIfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange AffineIfOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &AffineIfOpAdaptor::thenRegion() {
  return *odsRegions[0];
}

::mlir::Region &AffineIfOpAdaptor::elseRegion() {
  return *odsRegions[1];
}

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

::mlir::Operation::result_range AffineIfOp::results() {
  return getODSResults(0);
}

::mlir::Region &AffineIfOp::thenRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &AffineIfOp::elseRegion() {
  return (*this)->getRegion(1);
}

::mlir::LogicalResult AffineIfOp::verifyInvariants() {
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps0(*this, region, "thenRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps1(*this, region, "elseRegion", index++)))
        return ::mlir::failure();
  }
  return verify();
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineIfOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineLoadOp definitions
//===----------------------------------------------------------------------===//

AffineLoadOpAdaptor::AffineLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineLoadOpAdaptor::AffineLoadOpAdaptor(AffineLoadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffineLoadOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange AffineLoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr AffineLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

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

::mlir::Value AffineLoadOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range AffineLoadOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffineLoadOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineLoadOp::indicesMutable() {
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

::mlir::Value AffineLoadOp::result() {
  return *getODSResults(0).begin();
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

::mlir::LogicalResult AffineLoadOp::verifyInvariants() {
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
  return verify();
}

void AffineLoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineLoadOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineMaxOp definitions
//===----------------------------------------------------------------------===//

AffineMaxOpAdaptor::AffineMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineMaxOpAdaptor::AffineMaxOpAdaptor(AffineMaxOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AffineMaxOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr AffineMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr AffineMaxOpAdaptor::mapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineMaxOpAdaptor::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

::mlir::LogicalResult AffineMaxOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_map = odsAttrs.get("map");
    if (!tblgen_map)
      return emitError(loc, "'affine.max' op ""requires attribute 'map'");

    if (tblgen_map && !((tblgen_map.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'affine.max' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  }
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

::mlir::Operation::operand_range AffineMaxOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineMaxOp::operandsMutable() {
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

::mlir::AffineMapAttr AffineMaxOp::mapAttr() {
  return (*this)->getAttr(mapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineMaxOp::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

void AffineMaxOp::mapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(mapAttrName(), attr);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap affineMap, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(), affineMap, mapOperands);
    
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), map);
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineMaxOp::verifyInvariants() {
  {
    auto tblgen_map = (*this)->getAttr(mapAttrName());
    if (!tblgen_map)
      return emitOpError("requires attribute 'map'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_map, "map")))
      return ::mlir::failure();
  }
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
  return verify();
}

void AffineMaxOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineMaxOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineMinOp definitions
//===----------------------------------------------------------------------===//

AffineMinOpAdaptor::AffineMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineMinOpAdaptor::AffineMinOpAdaptor(AffineMinOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AffineMinOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr AffineMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr AffineMinOpAdaptor::mapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineMinOpAdaptor::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

::mlir::LogicalResult AffineMinOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_map = odsAttrs.get("map");
    if (!tblgen_map)
      return emitError(loc, "'affine.min' op ""requires attribute 'map'");

    if (tblgen_map && !((tblgen_map.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'affine.min' op ""attribute 'map' failed to satisfy constraint: AffineMap attribute");
  }
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

::mlir::Operation::operand_range AffineMinOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineMinOp::operandsMutable() {
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

::mlir::AffineMapAttr AffineMinOp::mapAttr() {
  return (*this)->getAttr(mapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineMinOp::map() {
  auto attr = mapAttr();
  return attr.getValue();
}

void AffineMinOp::mapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(mapAttrName(), attr);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, AffineMap affineMap, ValueRange mapOperands) {
      build(odsBuilder, odsState, odsBuilder.getIndexType(), affineMap, mapOperands);
    
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), map);
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMapAttr map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), map);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  odsState.addTypes(resultType0);
}

void AffineMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::AffineMap map, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(mapAttrName(odsState.name), ::mlir::AffineMapAttr::get(map));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffineMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AffineMinOp::verifyInvariants() {
  {
    auto tblgen_map = (*this)->getAttr(mapAttrName());
    if (!tblgen_map)
      return emitOpError("requires attribute 'map'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_map, "map")))
      return ::mlir::failure();
  }
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
  return verify();
}

void AffineMinOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineMinOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineParallelOp definitions
//===----------------------------------------------------------------------===//

AffineParallelOpAdaptor::AffineParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineParallelOpAdaptor::AffineParallelOpAdaptor(AffineParallelOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineParallelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineParallelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineParallelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AffineParallelOpAdaptor::mapOperands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr AffineParallelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr AffineParallelOpAdaptor::reductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("reductions").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AffineParallelOpAdaptor::reductions() {
  auto attr = reductionsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOpAdaptor::lowerBoundsMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("lowerBoundsMap").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineParallelOpAdaptor::lowerBoundsMap() {
  auto attr = lowerBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOpAdaptor::lowerBoundsGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("lowerBoundsGroups").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr AffineParallelOpAdaptor::lowerBoundsGroups() {
  auto attr = lowerBoundsGroupsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOpAdaptor::upperBoundsMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("upperBoundsMap").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap AffineParallelOpAdaptor::upperBoundsMap() {
  auto attr = upperBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOpAdaptor::upperBoundsGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("upperBoundsGroups").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr AffineParallelOpAdaptor::upperBoundsGroups() {
  auto attr = upperBoundsGroupsAttr();
  return attr;
}

::mlir::ArrayAttr AffineParallelOpAdaptor::stepsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("steps").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr AffineParallelOpAdaptor::steps() {
  auto attr = stepsAttr();
  return attr;
}

::mlir::RegionRange AffineParallelOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &AffineParallelOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult AffineParallelOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_reductions = odsAttrs.get("reductions");
    if (!tblgen_reductions)
      return emitError(loc, "'affine.parallel' op ""requires attribute 'reductions'");

    if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::arith::AtomicRMWKindAttr>()); }))))
      return emitError(loc, "'affine.parallel' op ""attribute 'reductions' failed to satisfy constraint: Reduction ops");
  }
  {
    auto tblgen_lowerBoundsMap = odsAttrs.get("lowerBoundsMap");
    if (!tblgen_lowerBoundsMap)
      return emitError(loc, "'affine.parallel' op ""requires attribute 'lowerBoundsMap'");

    if (tblgen_lowerBoundsMap && !((tblgen_lowerBoundsMap.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'affine.parallel' op ""attribute 'lowerBoundsMap' failed to satisfy constraint: AffineMap attribute");
  }
  {
    auto tblgen_lowerBoundsGroups = odsAttrs.get("lowerBoundsGroups");
    if (!tblgen_lowerBoundsGroups)
      return emitError(loc, "'affine.parallel' op ""requires attribute 'lowerBoundsGroups'");

    if (tblgen_lowerBoundsGroups && !(((tblgen_lowerBoundsGroups.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_lowerBoundsGroups.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
      return emitError(loc, "'affine.parallel' op ""attribute 'lowerBoundsGroups' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  {
    auto tblgen_upperBoundsMap = odsAttrs.get("upperBoundsMap");
    if (!tblgen_upperBoundsMap)
      return emitError(loc, "'affine.parallel' op ""requires attribute 'upperBoundsMap'");

    if (tblgen_upperBoundsMap && !((tblgen_upperBoundsMap.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'affine.parallel' op ""attribute 'upperBoundsMap' failed to satisfy constraint: AffineMap attribute");
  }
  {
    auto tblgen_upperBoundsGroups = odsAttrs.get("upperBoundsGroups");
    if (!tblgen_upperBoundsGroups)
      return emitError(loc, "'affine.parallel' op ""requires attribute 'upperBoundsGroups'");

    if (tblgen_upperBoundsGroups && !(((tblgen_upperBoundsGroups.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_upperBoundsGroups.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
      return emitError(loc, "'affine.parallel' op ""attribute 'upperBoundsGroups' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  {
    auto tblgen_steps = odsAttrs.get("steps");
    if (!tblgen_steps)
      return emitError(loc, "'affine.parallel' op ""requires attribute 'steps'");

    if (tblgen_steps && !(((tblgen_steps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_steps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'affine.parallel' op ""attribute 'steps' failed to satisfy constraint: 64-bit integer array attribute");
  }
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

::mlir::Operation::operand_range AffineParallelOp::mapOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineParallelOp::mapOperandsMutable() {
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

::mlir::Operation::result_range AffineParallelOp::results() {
  return getODSResults(0);
}

::mlir::Region &AffineParallelOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr AffineParallelOp::reductionsAttr() {
  return (*this)->getAttr(reductionsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AffineParallelOp::reductions() {
  auto attr = reductionsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOp::lowerBoundsMapAttr() {
  return (*this)->getAttr(lowerBoundsMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineParallelOp::lowerBoundsMap() {
  auto attr = lowerBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOp::lowerBoundsGroupsAttr() {
  return (*this)->getAttr(lowerBoundsGroupsAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr AffineParallelOp::lowerBoundsGroups() {
  auto attr = lowerBoundsGroupsAttr();
  return attr;
}

::mlir::AffineMapAttr AffineParallelOp::upperBoundsMapAttr() {
  return (*this)->getAttr(upperBoundsMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap AffineParallelOp::upperBoundsMap() {
  auto attr = upperBoundsMapAttr();
  return attr.getValue();
}

::mlir::DenseIntElementsAttr AffineParallelOp::upperBoundsGroupsAttr() {
  return (*this)->getAttr(upperBoundsGroupsAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr AffineParallelOp::upperBoundsGroups() {
  auto attr = upperBoundsGroupsAttr();
  return attr;
}

::mlir::ArrayAttr AffineParallelOp::stepsAttr() {
  return (*this)->getAttr(stepsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr AffineParallelOp::steps() {
  auto attr = stepsAttr();
  return attr;
}

void AffineParallelOp::reductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(reductionsAttrName(), attr);
}

void AffineParallelOp::lowerBoundsMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(lowerBoundsMapAttrName(), attr);
}

void AffineParallelOp::lowerBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(lowerBoundsGroupsAttrName(), attr);
}

void AffineParallelOp::upperBoundsMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(upperBoundsMapAttrName(), attr);
}

void AffineParallelOp::upperBoundsGroupsAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(upperBoundsGroupsAttrName(), attr);
}

void AffineParallelOp::stepsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stepsAttrName(), attr);
}

void AffineParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMapAttr lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMapAttr upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::mlir::ArrayAttr steps, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  odsState.addAttribute(lowerBoundsMapAttrName(odsState.name), lowerBoundsMap);
  odsState.addAttribute(lowerBoundsGroupsAttrName(odsState.name), lowerBoundsGroups);
  odsState.addAttribute(upperBoundsMapAttrName(odsState.name), upperBoundsMap);
  odsState.addAttribute(upperBoundsGroupsAttrName(odsState.name), upperBoundsGroups);
  odsState.addAttribute(stepsAttrName(odsState.name), steps);
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AffineParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ArrayAttr reductions, ::mlir::AffineMap lowerBoundsMap, ::mlir::DenseIntElementsAttr lowerBoundsGroups, ::mlir::AffineMap upperBoundsMap, ::mlir::DenseIntElementsAttr upperBoundsGroups, ::mlir::ArrayAttr steps, ::mlir::ValueRange mapOperands) {
  odsState.addOperands(mapOperands);
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  odsState.addAttribute(lowerBoundsMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(lowerBoundsMap));
  odsState.addAttribute(lowerBoundsGroupsAttrName(odsState.name), lowerBoundsGroups);
  odsState.addAttribute(upperBoundsMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(upperBoundsMap));
  odsState.addAttribute(upperBoundsGroupsAttrName(odsState.name), upperBoundsGroups);
  odsState.addAttribute(stepsAttrName(odsState.name), steps);
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

::mlir::LogicalResult AffineParallelOp::verifyInvariants() {
  {
    auto tblgen_reductions = (*this)->getAttr(reductionsAttrName());
    if (!tblgen_reductions)
      return emitOpError("requires attribute 'reductions'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps1(*this, tblgen_reductions, "reductions")))
      return ::mlir::failure();
  }
  {
    auto tblgen_lowerBoundsMap = (*this)->getAttr(lowerBoundsMapAttrName());
    if (!tblgen_lowerBoundsMap)
      return emitOpError("requires attribute 'lowerBoundsMap'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_lowerBoundsMap, "lowerBoundsMap")))
      return ::mlir::failure();
  }
  {
    auto tblgen_lowerBoundsGroups = (*this)->getAttr(lowerBoundsGroupsAttrName());
    if (!tblgen_lowerBoundsGroups)
      return emitOpError("requires attribute 'lowerBoundsGroups'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps2(*this, tblgen_lowerBoundsGroups, "lowerBoundsGroups")))
      return ::mlir::failure();
  }
  {
    auto tblgen_upperBoundsMap = (*this)->getAttr(upperBoundsMapAttrName());
    if (!tblgen_upperBoundsMap)
      return emitOpError("requires attribute 'upperBoundsMap'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps0(*this, tblgen_upperBoundsMap, "upperBoundsMap")))
      return ::mlir::failure();
  }
  {
    auto tblgen_upperBoundsGroups = (*this)->getAttr(upperBoundsGroupsAttrName());
    if (!tblgen_upperBoundsGroups)
      return emitOpError("requires attribute 'upperBoundsGroups'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps2(*this, tblgen_upperBoundsGroups, "upperBoundsGroups")))
      return ::mlir::failure();
  }
  {
    auto tblgen_steps = (*this)->getAttr(stepsAttrName());
    if (!tblgen_steps)
      return emitOpError("requires attribute 'steps'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps3(*this, tblgen_steps, "steps")))
      return ::mlir::failure();
  }
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

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_AffineOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return verify();
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineParallelOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffinePrefetchOp definitions
//===----------------------------------------------------------------------===//

AffinePrefetchOpAdaptor::AffinePrefetchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffinePrefetchOpAdaptor::AffinePrefetchOpAdaptor(AffinePrefetchOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffinePrefetchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffinePrefetchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffinePrefetchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffinePrefetchOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange AffinePrefetchOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr AffinePrefetchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr AffinePrefetchOpAdaptor::isWriteAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("isWrite").cast<::mlir::BoolAttr>();
  return attr;
}

bool AffinePrefetchOpAdaptor::isWrite() {
  auto attr = isWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr AffinePrefetchOpAdaptor::localityHintAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("localityHint").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t AffinePrefetchOpAdaptor::localityHint() {
  auto attr = localityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr AffinePrefetchOpAdaptor::isDataCacheAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("isDataCache").cast<::mlir::BoolAttr>();
  return attr;
}

bool AffinePrefetchOpAdaptor::isDataCache() {
  auto attr = isDataCacheAttr();
  return attr.getValue();
}

::mlir::LogicalResult AffinePrefetchOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_isWrite = odsAttrs.get("isWrite");
    if (!tblgen_isWrite)
      return emitError(loc, "'affine.prefetch' op ""requires attribute 'isWrite'");

    if (tblgen_isWrite && !((tblgen_isWrite.isa<::mlir::BoolAttr>())))
      return emitError(loc, "'affine.prefetch' op ""attribute 'isWrite' failed to satisfy constraint: bool attribute");
  }
  {
    auto tblgen_localityHint = odsAttrs.get("localityHint");
    if (!tblgen_localityHint)
      return emitError(loc, "'affine.prefetch' op ""requires attribute 'localityHint'");

    if (tblgen_localityHint && !((((tblgen_localityHint.isa<::mlir::IntegerAttr>())) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() <= 3))))
      return emitError(loc, "'affine.prefetch' op ""attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");
  }
  {
    auto tblgen_isDataCache = odsAttrs.get("isDataCache");
    if (!tblgen_isDataCache)
      return emitError(loc, "'affine.prefetch' op ""requires attribute 'isDataCache'");

    if (tblgen_isDataCache && !((tblgen_isDataCache.isa<::mlir::BoolAttr>())))
      return emitError(loc, "'affine.prefetch' op ""attribute 'isDataCache' failed to satisfy constraint: bool attribute");
  }
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

::mlir::Value AffinePrefetchOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range AffinePrefetchOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffinePrefetchOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffinePrefetchOp::indicesMutable() {
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

::mlir::BoolAttr AffinePrefetchOp::isWriteAttr() {
  return (*this)->getAttr(isWriteAttrName()).cast<::mlir::BoolAttr>();
}

bool AffinePrefetchOp::isWrite() {
  auto attr = isWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr AffinePrefetchOp::localityHintAttr() {
  return (*this)->getAttr(localityHintAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t AffinePrefetchOp::localityHint() {
  auto attr = localityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr AffinePrefetchOp::isDataCacheAttr() {
  return (*this)->getAttr(isDataCacheAttrName()).cast<::mlir::BoolAttr>();
}

bool AffinePrefetchOp::isDataCache() {
  auto attr = isDataCacheAttr();
  return attr.getValue();
}

void AffinePrefetchOp::isWriteAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(isWriteAttrName(), attr);
}

void AffinePrefetchOp::localityHintAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(localityHintAttrName(), attr);
}

void AffinePrefetchOp::isDataCacheAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(isDataCacheAttrName(), attr);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, AffineMap map, ArrayRef<Value> mapOperands, bool isWrite, unsigned localityHint, bool isDataCache) {
      assert(map.getNumInputs() == mapOperands.size()
             && "inconsistent index info");
      auto localityHintAttr = odsBuilder.getI32IntegerAttr(localityHint);
      auto isWriteAttr = odsBuilder.getBoolAttr(isWrite);
      auto isDataCacheAttr = odsBuilder.getBoolAttr(isDataCache);
      odsState.addOperands(memref);
      odsState.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
      odsState.addOperands(mapOperands);
      odsState.addAttribute(getLocalityHintAttrName(), localityHintAttr);
      odsState.addAttribute(getIsWriteAttrName(), isWriteAttr);
      odsState.addAttribute(getIsDataCacheAttrName(), isDataCacheAttr);
    
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(isWriteAttrName(odsState.name), isWrite);
  odsState.addAttribute(localityHintAttrName(odsState.name), localityHint);
  odsState.addAttribute(isDataCacheAttrName(odsState.name), isDataCache);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(isWriteAttrName(odsState.name), isWrite);
  odsState.addAttribute(localityHintAttrName(odsState.name), localityHint);
  odsState.addAttribute(isDataCacheAttrName(odsState.name), isDataCache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(isWriteAttrName(odsState.name), odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute(localityHintAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute(isDataCacheAttrName(odsState.name), odsBuilder.getBoolAttr(isDataCache));
}

void AffinePrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(isWriteAttrName(odsState.name), odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute(localityHintAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute(isDataCacheAttrName(odsState.name), odsBuilder.getBoolAttr(isDataCache));
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

::mlir::LogicalResult AffinePrefetchOp::verifyInvariants() {
  {
    auto tblgen_isWrite = (*this)->getAttr(isWriteAttrName());
    if (!tblgen_isWrite)
      return emitOpError("requires attribute 'isWrite'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(*this, tblgen_isWrite, "isWrite")))
      return ::mlir::failure();
  }
  {
    auto tblgen_localityHint = (*this)->getAttr(localityHintAttrName());
    if (!tblgen_localityHint)
      return emitOpError("requires attribute 'localityHint'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps5(*this, tblgen_localityHint, "localityHint")))
      return ::mlir::failure();
  }
  {
    auto tblgen_isDataCache = (*this)->getAttr(isDataCacheAttrName());
    if (!tblgen_isDataCache)
      return emitOpError("requires attribute 'isDataCache'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_AffineOps4(*this, tblgen_isDataCache, "isDataCache")))
      return ::mlir::failure();
  }
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
  return verify();
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffinePrefetchOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineStoreOp definitions
//===----------------------------------------------------------------------===//

AffineStoreOpAdaptor::AffineStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineStoreOpAdaptor::AffineStoreOpAdaptor(AffineStoreOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffineStoreOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value AffineStoreOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange AffineStoreOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr AffineStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

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

::mlir::Value AffineStoreOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value AffineStoreOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range AffineStoreOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AffineStoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineStoreOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineStoreOp::indicesMutable() {
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

::mlir::LogicalResult AffineStoreOp::verifyInvariants() {
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
  return verify();
}

void AffineStoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineStoreOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineVectorLoadOp definitions
//===----------------------------------------------------------------------===//

AffineVectorLoadOpAdaptor::AffineVectorLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineVectorLoadOpAdaptor::AffineVectorLoadOpAdaptor(AffineVectorLoadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineVectorLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineVectorLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineVectorLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffineVectorLoadOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange AffineVectorLoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr AffineVectorLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

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

::mlir::Value AffineVectorLoadOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range AffineVectorLoadOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AffineVectorLoadOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineVectorLoadOp::indicesMutable() {
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

::mlir::Value AffineVectorLoadOp::result() {
  return *getODSResults(0).begin();
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

::mlir::LogicalResult AffineVectorLoadOp::verifyInvariants() {
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
  return verify();
}

void AffineVectorLoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineVectorLoadOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineVectorStoreOp definitions
//===----------------------------------------------------------------------===//

AffineVectorStoreOpAdaptor::AffineVectorStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineVectorStoreOpAdaptor::AffineVectorStoreOpAdaptor(AffineVectorStoreOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineVectorStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineVectorStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineVectorStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AffineVectorStoreOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value AffineVectorStoreOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange AffineVectorStoreOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr AffineVectorStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

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

::mlir::Value AffineVectorStoreOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value AffineVectorStoreOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range AffineVectorStoreOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AffineVectorStoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineVectorStoreOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AffineVectorStoreOp::indicesMutable() {
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

::mlir::LogicalResult AffineVectorStoreOp::verifyInvariants() {
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
  return verify();
}

void AffineVectorStoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineVectorStoreOp)

namespace mlir {

//===----------------------------------------------------------------------===//
// mlir::AffineYieldOp definitions
//===----------------------------------------------------------------------===//

AffineYieldOpAdaptor::AffineYieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AffineYieldOpAdaptor::AffineYieldOpAdaptor(AffineYieldOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AffineYieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AffineYieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange AffineYieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange AffineYieldOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr AffineYieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

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

::mlir::Operation::operand_range AffineYieldOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AffineYieldOp::operandsMutable() {
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
 build(odsBuilder, odsState, llvm::None); 
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

::mlir::LogicalResult AffineYieldOp::verifyInvariants() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AffineOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return verify();
}

::mlir::ParseResult AffineYieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> operandsTypes;
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  _odsPrinter << ' ';
  _odsPrinter << operands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << operands().getTypes();
  }
}

void AffineYieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(mlir::AffineYieldOp)


#endif  // GET_OP_CLASSES

