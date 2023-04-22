/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tensor::CastOp,
::mlir::tensor::CollapseShapeOp,
::mlir::tensor::DimOp,
::mlir::tensor::EmptyOp,
::mlir::tensor::ExpandShapeOp,
::mlir::tensor::ExtractOp,
::mlir::tensor::ExtractSliceOp,
::mlir::tensor::FromElementsOp,
::mlir::tensor::GatherOp,
::mlir::tensor::GenerateOp,
::mlir::tensor::InsertOp,
::mlir::tensor::InsertSliceOp,
::mlir::tensor::PackOp,
::mlir::tensor::PadOp,
::mlir::tensor::ParallelInsertSliceOp,
::mlir::tensor::RankOp,
::mlir::tensor::ReshapeOp,
::mlir::tensor::ScatterOp,
::mlir::tensor::SplatOp,
::mlir::tensor::UnPackOp,
::mlir::tensor::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tensor {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::TensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank() >= 1)))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be non-0-ranked or unranked tensor, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::RankedTensorType>())) && ((type.cast<::mlir::ShapedType>().hasStaticShape()))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be statically shaped tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of signless integer or index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::RankedTensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isa<::mlir::IndexType>())); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D tensor of signless integer or index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_TensorOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger())) || ((type.isa<::mlir::IndexType>())) || ((type.isa<::mlir::FloatType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer/index/float type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TensorOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Array of 64-bit integer array attributes";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TensorOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseI64ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i64 dense array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_TensorOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_TensorOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::CastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CastOpGenericAdaptorBase::CastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.cast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CastOpAdaptor::CastOpAdaptor(CastOp op) : CastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CastOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CastOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CastOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(dest);
}

void CastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addTypes(destTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::CastOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::CollapseShapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CollapseShapeOpGenericAdaptorBase::CollapseShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.collapse_shape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CollapseShapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CollapseShapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CollapseShapeOpGenericAdaptorBase::getReassociationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CollapseShapeOp::getReassociationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CollapseShapeOpGenericAdaptorBase::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

} // namespace detail
CollapseShapeOpAdaptor::CollapseShapeOpAdaptor(CollapseShapeOp op) : CollapseShapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CollapseShapeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.collapse_shape' op ""requires attribute 'reassociation'");
    if (namedAttrIt->getName() == CollapseShapeOp::getReassociationAttrName(*odsOpName)) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_reassociation && !(((tblgen_reassociation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reassociation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); }))))
    return emitError(loc, "'tensor.collapse_shape' op ""attribute 'reassociation' failed to satisfy constraint: Array of 64-bit integer array attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CollapseShapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CollapseShapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> CollapseShapeOp::getSrc() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CollapseShapeOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CollapseShapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CollapseShapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> CollapseShapeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::ArrayAttr CollapseShapeOp::getReassociationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReassociationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CollapseShapeOp::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

void CollapseShapeOp::setReassociationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReassociationAttrName(), attr);
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs) {
      auto reassociationMaps =
          convertReassociationMapsToIndices(odsBuilder, reassociation);
      build(odsBuilder, odsState, src, reassociationMaps, attrs);
    
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState, resultType, src, attrs);
      odsState.addAttribute("reassociation",
          getReassociationIndicesAttribute(odsBuilder, reassociation));
    
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs) {
      auto reassociationMaps =
          convertReassociationMapsToIndices(odsBuilder, reassociation);
      build(odsBuilder, odsState, resultType, src, reassociationMaps, attrs);
    
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  odsState.addTypes(result);
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CollapseShapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CollapseShapeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'reassociation'");
    if (namedAttrIt->getName() == getReassociationAttrName()) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps0(*this, tblgen_reassociation, "reassociation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CollapseShapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CollapseShapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::ArrayAttr reassociationAttr;
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(reassociationAttr, parser.getBuilder().getType<::mlir::NoneType>(), "reassociation",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CollapseShapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getReassociationAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("reassociation");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CollapseShapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::CollapseShapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::DimOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DimOpGenericAdaptorBase::DimOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.dim", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DimOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr DimOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
DimOpAdaptor::DimOpAdaptor(DimOp op) : DimOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DimOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DimOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DimOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> DimOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> DimOp::getIndex() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange DimOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DimOp::getIndexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DimOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DimOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> DimOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);
  odsState.addTypes(result);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DimOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(DimOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult DimOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DimOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult DimOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult DimOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, odsBuildableType0, indexOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DimOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getIndex();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void DimOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::DimOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::EmptyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EmptyOpGenericAdaptorBase::EmptyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.empty", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EmptyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr EmptyOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
EmptyOpAdaptor::EmptyOpAdaptor(EmptyOp op) : EmptyOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult EmptyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EmptyOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range EmptyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range EmptyOp::getDynamicSizes() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange EmptyOp::getDynamicSizesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> EmptyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EmptyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> EmptyOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

void EmptyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamicSizes) {
  odsState.addOperands(dynamicSizes);
  odsState.addTypes(result);
}

void EmptyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EmptyOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EmptyOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult EmptyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dynamicSizesOperands;
  ::llvm::SMLoc dynamicSizesOperandsLoc;
  (void)dynamicSizesOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  dynamicSizesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dynamicSizesOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(dynamicSizesOperands, odsBuildableType0, dynamicSizesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EmptyOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getDynamicSizes();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void EmptyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::EmptyOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExpandShapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExpandShapeOpGenericAdaptorBase::ExpandShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.expand_shape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExpandShapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExpandShapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExpandShapeOpGenericAdaptorBase::getReassociationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExpandShapeOp::getReassociationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExpandShapeOpGenericAdaptorBase::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

} // namespace detail
ExpandShapeOpAdaptor::ExpandShapeOpAdaptor(ExpandShapeOp op) : ExpandShapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExpandShapeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.expand_shape' op ""requires attribute 'reassociation'");
    if (namedAttrIt->getName() == ExpandShapeOp::getReassociationAttrName(*odsOpName)) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_reassociation && !(((tblgen_reassociation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reassociation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); }))))
    return emitError(loc, "'tensor.expand_shape' op ""attribute 'reassociation' failed to satisfy constraint: Array of 64-bit integer array attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExpandShapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpandShapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ExpandShapeOp::getSrc() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExpandShapeOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExpandShapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpandShapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ExpandShapeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::ArrayAttr ExpandShapeOp::getReassociationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReassociationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExpandShapeOp::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

void ExpandShapeOp::setReassociationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReassociationAttrName(), attr);
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState, resultType, src, attrs);
      odsState.addAttribute("reassociation",
          getReassociationIndicesAttribute(odsBuilder, reassociation));
    
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs) {
      auto reassociationMaps =
          convertReassociationMapsToIndices(odsBuilder, reassociation);
      build(odsBuilder, odsState, resultType, src, reassociationMaps, attrs);
    
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  odsState.addTypes(result);
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpandShapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExpandShapeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'reassociation'");
    if (namedAttrIt->getName() == getReassociationAttrName()) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps0(*this, tblgen_reassociation, "reassociation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExpandShapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExpandShapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::ArrayAttr reassociationAttr;
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(reassociationAttr, parser.getBuilder().getType<::mlir::NoneType>(), "reassociation",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExpandShapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getReassociationAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("reassociation");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExpandShapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ExpandShapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExtractOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractOpGenericAdaptorBase::ExtractOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.extract", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ExtractOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ExtractOpAdaptor::ExtractOpAdaptor(ExtractOp op) : ExtractOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ExtractOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ExtractOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range ExtractOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ExtractOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::ValueRange indices) {
  odsState.addOperands(tensor);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::ValueRange indices) {
  odsState.addOperands(tensor);
  odsState.addOperands(indices);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ExtractOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::ValueRange indices) {
  odsState.addOperands(tensor);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ExtractOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ExtractOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType().cast<TensorType>().getElementType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches element type of tensor");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult ExtractOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType().cast<TensorType>().getElementType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ExtractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  for (::mlir::Type type : tensorTypes) {
    (void)type;
    if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'tensor' must be ranked tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(tensorTypes[0].cast<TensorType>().getElementType());
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExtractOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ExtractOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExtractSliceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractSliceOpGenericAdaptorBase::ExtractSliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.extract_slice", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractSliceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ExtractSliceOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ExtractSliceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ExtractSliceOpGenericAdaptorBase::getStaticOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, ExtractSliceOp::getStaticOffsetsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ExtractSliceOpGenericAdaptorBase::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ExtractSliceOpGenericAdaptorBase::getStaticSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ExtractSliceOp::getStaticSizesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ExtractSliceOpGenericAdaptorBase::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ExtractSliceOpGenericAdaptorBase::getStaticStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, ExtractSliceOp::getStaticStridesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ExtractSliceOpGenericAdaptorBase::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

} // namespace detail
ExtractSliceOpAdaptor::ExtractSliceOpAdaptor(ExtractSliceOp op) : ExtractSliceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractSliceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ExtractSliceOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == ExtractSliceOp::getStaticOffsetsAttrName(*odsOpName)) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == ExtractSliceOp::getStaticSizesAttrName(*odsOpName)) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.extract_slice' op ""requires attribute 'static_strides'");
    if (namedAttrIt->getName() == ExtractSliceOp::getStaticStridesAttrName(*odsOpName)) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'tensor.extract_slice' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_offsets && !((tblgen_static_offsets.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.extract_slice' op ""attribute 'static_offsets' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_sizes && !((tblgen_static_sizes.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.extract_slice' op ""attribute 'static_sizes' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_strides && !((tblgen_static_strides.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.extract_slice' op ""attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractSliceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ExtractSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ExtractSliceOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range ExtractSliceOp::getOffsets() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ExtractSliceOp::getSizes() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ExtractSliceOp::getStrides() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange ExtractSliceOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExtractSliceOp::getOffsetsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExtractSliceOp::getSizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExtractSliceOp::getStridesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ExtractSliceOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr ExtractSliceOp::getStaticOffsetsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getStaticOffsetsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ExtractSliceOp::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ExtractSliceOp::getStaticSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getStaticSizesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ExtractSliceOp::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ExtractSliceOp::getStaticStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getStaticStridesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ExtractSliceOp::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

void ExtractSliceOp::setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticOffsetsAttrName(), attr);
}

void ExtractSliceOp::setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticOffsetsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ExtractSliceOp::setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticSizesAttrName(), attr);
}

void ExtractSliceOp::setStaticSizes(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticSizesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ExtractSliceOp::setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticStridesAttrName(), attr);
}

void ExtractSliceOp::setStaticStrides(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticStridesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ExtractSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  odsState.addTypes(result);
}

void ExtractSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  odsState.addTypes(result);
}

void ExtractSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractSliceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == getStaticOffsetsAttrName()) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == getStaticSizesAttrName()) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_strides'");
    if (namedAttrIt->getName() == getStaticStridesAttrName()) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_offsets, "static_offsets")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_sizes, "static_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_strides, "static_strides")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractSliceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExtractSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_offsetsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_sizesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, offsetsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, stridesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractSliceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  printDynamicIndexList(_odsPrinter, *this, getOffsets(), getStaticOffsetsAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getSizes(), getStaticSizesAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getStrides(), getStaticStridesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("static_offsets");
  elidedAttrs.push_back("static_sizes");
  elidedAttrs.push_back("static_strides");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExtractSliceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ExtractSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::FromElementsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FromElementsOpGenericAdaptorBase::FromElementsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.from_elements", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FromElementsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr FromElementsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FromElementsOpAdaptor::FromElementsOpAdaptor(FromElementsOp op) : FromElementsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FromElementsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FromElementsOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range FromElementsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FromElementsOp::getElements() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange FromElementsOp::getElementsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FromElementsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FromElementsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> FromElementsOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult FromElementsOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(SmallVector<Type, 2>((*this->getODSResults(0).begin()).getType().cast<RankedTensorType>().getNumElements(), (*this->getODSResults(0).begin()).getType().cast<RankedTensorType>().getElementType()), this->getODSOperands(0).getType()))))
    return emitOpError("failed to verify that operand types match result element type");
  return ::mlir::success();
}

::mlir::LogicalResult FromElementsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FromElementsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> elementsOperands;
  ::llvm::SMLoc elementsOperandsLoc;
  (void)elementsOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  elementsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(elementsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!((((type.isa<::mlir::RankedTensorType>())) && ((type.cast<::mlir::ShapedType>().hasStaticShape()))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be statically shaped tensor of any type values, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(elementsOperands, SmallVector<Type, 2>(resultTypes[0].cast<RankedTensorType>().getNumElements(), resultTypes[0].cast<RankedTensorType>().getElementType()), elementsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FromElementsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getElements();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FromElementsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::FromElementsOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::GatherOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GatherOpGenericAdaptorBase::GatherOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.gather", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GatherOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GatherOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr GatherOpGenericAdaptorBase::getGatherDimsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GatherOp::getGatherDimsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> GatherOpGenericAdaptorBase::getGatherDims() {
  auto attr = getGatherDimsAttr();
  return attr;
}

::mlir::UnitAttr GatherOpGenericAdaptorBase::getUniqueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GatherOp::getUniqueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GatherOpGenericAdaptorBase::getUnique() {
  auto attr = getUniqueAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
GatherOpAdaptor::GatherOpAdaptor(GatherOp op) : GatherOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GatherOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_gather_dims;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.gather' op ""requires attribute 'gather_dims'");
    if (namedAttrIt->getName() == GatherOp::getGatherDimsAttrName(*odsOpName)) {
      tblgen_gather_dims = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_unique;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GatherOp::getUniqueAttrName(*odsOpName)) {
      tblgen_unique = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_gather_dims && !((tblgen_gather_dims.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.gather' op ""attribute 'gather_dims' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_unique && !((tblgen_unique.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'tensor.gather' op ""attribute 'unique' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GatherOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GatherOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> GatherOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> GatherOp::getIndices() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange GatherOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GatherOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GatherOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> GatherOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr GatherOp::getGatherDimsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGatherDimsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> GatherOp::getGatherDims() {
  auto attr = getGatherDimsAttr();
  return attr;
}

::mlir::UnitAttr GatherOp::getUniqueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getUniqueAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GatherOp::getUnique() {
  auto attr = getUniqueAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void GatherOp::setGatherDimsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getGatherDimsAttrName(), attr);
}

void GatherOp::setGatherDims(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getGatherDimsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void GatherOp::setUniqueAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUniqueAttrName(), attr);
}

void GatherOp::setUnique(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getUniqueAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getUniqueAttrName());
}

::mlir::Attribute GatherOp::removeUniqueAttr() {
  return (*this)->removeAttr(getUniqueAttrName());
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr gather_dims, /*optional*/::mlir::UnitAttr unique) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addAttribute(getGatherDimsAttrName(odsState.name), gather_dims);
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), unique);
  }
  odsState.addTypes(result);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr gather_dims, /*optional*/::mlir::UnitAttr unique) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addAttribute(getGatherDimsAttrName(odsState.name), gather_dims);
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), unique);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> gather_dims, /*optional*/bool unique) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addAttribute(getGatherDimsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(gather_dims));
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), ((unique) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(result);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> gather_dims, /*optional*/bool unique) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addAttribute(getGatherDimsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(gather_dims));
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), ((unique) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GatherOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GatherOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_gather_dims;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'gather_dims'");
    if (namedAttrIt->getName() == getGatherDimsAttrName()) {
      tblgen_gather_dims = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_unique;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getUniqueAttrName()) {
      tblgen_unique = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_gather_dims, "gather_dims")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps2(*this, tblgen_unique, "unique")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GatherOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GatherOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand indicesRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> indicesOperands(indicesRawOperands);  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::DenseI64ArrayAttr gather_dimsAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indicesRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseKeyword("gather_dims"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(gather_dimsAttr, ::mlir::Type{}, "gather_dims",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("unique"))) {
    result.addAttribute("unique", parser.getBuilder().getUnitAttr());
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(sourceOperands, indicesOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GatherOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ' ' << "gather_dims";
  _odsPrinter << "(";
_odsPrinter.printStrippedAttrOrType(getGatherDimsAttr());
  _odsPrinter << ")";
  if ((*this)->getAttr("unique")) {
    _odsPrinter << ' ' << "unique";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("gather_dims");
  elidedAttrs.push_back("unique");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getUniqueAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("unique");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void GatherOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::GatherOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::GenerateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GenerateOpGenericAdaptorBase::GenerateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.generate", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GenerateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr GenerateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &GenerateOpGenericAdaptorBase::getBody() {
  return *odsRegions[0];
}

::mlir::RegionRange GenerateOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
GenerateOpAdaptor::GenerateOpAdaptor(GenerateOp op) : GenerateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GenerateOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GenerateOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GenerateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenerateOp::getDynamicExtents() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GenerateOp::getDynamicExtentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GenerateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenerateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> GenerateOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::Region &GenerateOp::getBody() {
  return (*this)->getRegion(0);
}

void GenerateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamicExtents) {
  odsState.addOperands(dynamicExtents);
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void GenerateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GenerateOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TensorOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GenerateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GenerateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dynamicExtentsOperands;
  ::llvm::SMLoc dynamicExtentsOperandsLoc;
  (void)dynamicExtentsOperandsLoc;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  dynamicExtentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dynamicExtentsOperands))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();

  ensureTerminator(*bodyRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addRegion(std::move(bodyRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(dynamicExtentsOperands, odsBuildableType0, dynamicExtentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GenerateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDynamicExtents();
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getBody().empty() ? nullptr : getBody().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(getBody(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::GenerateOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::InsertOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InsertOpGenericAdaptorBase::InsertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.insert", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InsertOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr InsertOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
InsertOpAdaptor::InsertOpAdaptor(InsertOp op) : InsertOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InsertOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InsertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOp::getScalar() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> InsertOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range InsertOp::getIndices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InsertOp::getScalarMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> InsertOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices) {
  odsState.addOperands(scalar);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices) {
  odsState.addOperands(scalar);
  odsState.addOperands(dest);
  odsState.addOperands(indices);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices) {
  odsState.addOperands(scalar);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches type of dest");
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<TensorType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that scalar type matches element type of dest");
  return ::mlir::success();
}

::mlir::LogicalResult InsertOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult InsertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand scalarRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> scalarOperands(scalarRawOperands);  ::llvm::SMLoc scalarOperandsLoc;
  (void)scalarOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  scalarOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(scalarRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  for (::mlir::Type type : destTypes) {
    (void)type;
    if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'dest' must be ranked tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(destTypes[0]);
  if (parser.resolveOperands(scalarOperands, destTypes[0].cast<TensorType>().getElementType(), scalarOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getScalar();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::InsertOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::InsertSliceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InsertSliceOpGenericAdaptorBase::InsertSliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.insert_slice", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InsertSliceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, InsertSliceOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr InsertSliceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr InsertSliceOpGenericAdaptorBase::getStaticOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, InsertSliceOp::getStaticOffsetsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> InsertSliceOpGenericAdaptorBase::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr InsertSliceOpGenericAdaptorBase::getStaticSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, InsertSliceOp::getStaticSizesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> InsertSliceOpGenericAdaptorBase::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr InsertSliceOpGenericAdaptorBase::getStaticStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, InsertSliceOp::getStaticStridesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> InsertSliceOpGenericAdaptorBase::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

} // namespace detail
InsertSliceOpAdaptor::InsertSliceOpAdaptor(InsertSliceOp op) : InsertSliceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InsertSliceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == InsertSliceOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == InsertSliceOp::getStaticOffsetsAttrName(*odsOpName)) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == InsertSliceOp::getStaticSizesAttrName(*odsOpName)) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.insert_slice' op ""requires attribute 'static_strides'");
    if (namedAttrIt->getName() == InsertSliceOp::getStaticStridesAttrName(*odsOpName)) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'tensor.insert_slice' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_offsets && !((tblgen_static_offsets.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.insert_slice' op ""attribute 'static_offsets' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_sizes && !((tblgen_static_sizes.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.insert_slice' op ""attribute 'static_sizes' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_strides && !((tblgen_static_strides.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.insert_slice' op ""attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertSliceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range InsertSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> InsertSliceOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> InsertSliceOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range InsertSliceOp::getOffsets() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range InsertSliceOp::getSizes() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range InsertSliceOp::getStrides() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange InsertSliceOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::getOffsetsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::getSizesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InsertSliceOp::getStridesMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> InsertSliceOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr InsertSliceOp::getStaticOffsetsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getStaticOffsetsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> InsertSliceOp::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr InsertSliceOp::getStaticSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getStaticSizesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> InsertSliceOp::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr InsertSliceOp::getStaticStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getStaticStridesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> InsertSliceOp::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

void InsertSliceOp::setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticOffsetsAttrName(), attr);
}

void InsertSliceOp::setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticOffsetsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void InsertSliceOp::setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticSizesAttrName(), attr);
}

void InsertSliceOp::setStaticSizes(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticSizesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void InsertSliceOp::setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticStridesAttrName(), attr);
}

void InsertSliceOp::setStaticStrides(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticStridesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  odsState.addTypes(result);
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertSliceOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  odsState.addTypes(result);
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertSliceOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertSliceOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertSliceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == getStaticOffsetsAttrName()) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == getStaticSizesAttrName()) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_strides'");
    if (namedAttrIt->getName() == getStaticStridesAttrName()) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_offsets, "static_offsets")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_sizes, "static_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_strides, "static_strides")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that expected result type to match dest type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertSliceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertSliceOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult InsertSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_offsetsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_sizesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  for (::mlir::Type type : destTypes) {
    (void)type;
    if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'dest' must be ranked tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(destTypes[0]);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, offsetsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, stridesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertSliceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  printDynamicIndexList(_odsPrinter, *this, getOffsets(), getStaticOffsetsAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getSizes(), getStaticSizesAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getStrides(), getStaticStridesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("static_offsets");
  elidedAttrs.push_back("static_sizes");
  elidedAttrs.push_back("static_strides");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertSliceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::InsertSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::PackOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PackOpGenericAdaptorBase::PackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.pack", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PackOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, PackOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PackOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr PackOpGenericAdaptorBase::getOuterDimsPermAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, PackOp::getOuterDimsPermAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseI64ArrayAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getDenseI64ArrayAttr({});
  return attr;
}

::llvm::ArrayRef<int64_t> PackOpGenericAdaptorBase::getOuterDimsPerm() {
  auto attr = getOuterDimsPermAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getDenseI64ArrayAttr({});
  return attr;
}

::mlir::DenseI64ArrayAttr PackOpGenericAdaptorBase::getInnerDimsPosAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, PackOp::getInnerDimsPosAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> PackOpGenericAdaptorBase::getInnerDimsPos() {
  auto attr = getInnerDimsPosAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr PackOpGenericAdaptorBase::getStaticInnerTilesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, PackOp::getStaticInnerTilesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> PackOpGenericAdaptorBase::getStaticInnerTiles() {
  auto attr = getStaticInnerTilesAttr();
  return attr;
}

} // namespace detail
PackOpAdaptor::PackOpAdaptor(PackOp op) : PackOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PackOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inner_dims_pos;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.pack' op ""requires attribute 'inner_dims_pos'");
    if (namedAttrIt->getName() == PackOp::getInnerDimsPosAttrName(*odsOpName)) {
      tblgen_inner_dims_pos = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.pack' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PackOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_inner_tiles;
  ::mlir::Attribute tblgen_outer_dims_perm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.pack' op ""requires attribute 'static_inner_tiles'");
    if (namedAttrIt->getName() == PackOp::getStaticInnerTilesAttrName(*odsOpName)) {
      tblgen_static_inner_tiles = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PackOp::getOuterDimsPermAttrName(*odsOpName)) {
      tblgen_outer_dims_perm = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'tensor.pack' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_outer_dims_perm && !((tblgen_outer_dims_perm.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.pack' op ""attribute 'outer_dims_perm' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_inner_dims_pos && !((tblgen_inner_dims_pos.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.pack' op ""attribute 'inner_dims_pos' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_inner_tiles && !((tblgen_static_inner_tiles.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.pack' op ""attribute 'static_inner_tiles' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PackOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PackOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> PackOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> PackOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::Value PackOp::getPaddingValue() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::Operation::operand_range PackOp::getInnerTiles() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange PackOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PackOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PackOp::getPaddingValueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PackOp::getInnerTilesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PackOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PackOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> PackOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr PackOp::getOuterDimsPermAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getOuterDimsPermAttrName()).dyn_cast_or_null<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> PackOp::getOuterDimsPerm() {
  auto attr = getOuterDimsPermAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr({});
  return attr;
}

::mlir::DenseI64ArrayAttr PackOp::getInnerDimsPosAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getInnerDimsPosAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> PackOp::getInnerDimsPos() {
  auto attr = getInnerDimsPosAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr PackOp::getStaticInnerTilesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStaticInnerTilesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> PackOp::getStaticInnerTiles() {
  auto attr = getStaticInnerTilesAttr();
  return attr;
}

void PackOp::setOuterDimsPermAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getOuterDimsPermAttrName(), attr);
}

void PackOp::setOuterDimsPerm(::std::optional<::llvm::ArrayRef<int64_t>> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOuterDimsPermAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(*attrValue));
    (*this)->removeAttr(getOuterDimsPermAttrName());
}

void PackOp::setInnerDimsPosAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getInnerDimsPosAttrName(), attr);
}

void PackOp::setInnerDimsPos(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getInnerDimsPosAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void PackOp::setStaticInnerTilesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticInnerTilesAttrName(), attr);
}

void PackOp::setStaticInnerTiles(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticInnerTilesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

::mlir::Attribute PackOp::removeOuterDimsPermAttr() {
  return (*this)->removeAttr(getOuterDimsPermAttrName());
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (padding_value)
    odsState.addOperands(padding_value);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, (padding_value ? 1 : 0), static_cast<int32_t>(inner_tiles.size())}));
  if (outer_dims_perm) {
    odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), outer_dims_perm);
  }
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), inner_dims_pos);
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), static_inner_tiles);
  odsState.addTypes(result);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (padding_value)
    odsState.addOperands(padding_value);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, (padding_value ? 1 : 0), static_cast<int32_t>(inner_tiles.size())}));
  if (outer_dims_perm) {
    odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), outer_dims_perm);
  }
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), inner_dims_pos);
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), static_inner_tiles);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(PackOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (padding_value)
    odsState.addOperands(padding_value);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, (padding_value ? 1 : 0), static_cast<int32_t>(inner_tiles.size())}));
  if (outer_dims_perm) {
    odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), outer_dims_perm);
  }
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), inner_dims_pos);
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), static_inner_tiles);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (padding_value)
    odsState.addOperands(padding_value);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, (padding_value ? 1 : 0), static_cast<int32_t>(inner_tiles.size())}));
  odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(outer_dims_perm));
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(inner_dims_pos));
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_inner_tiles));
  odsState.addTypes(result);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (padding_value)
    odsState.addOperands(padding_value);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, (padding_value ? 1 : 0), static_cast<int32_t>(inner_tiles.size())}));
  odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(outer_dims_perm));
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(inner_dims_pos));
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_inner_tiles));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(PackOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (padding_value)
    odsState.addOperands(padding_value);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, (padding_value ? 1 : 0), static_cast<int32_t>(inner_tiles.size())}));
  odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(outer_dims_perm));
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(inner_dims_pos));
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_inner_tiles));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(PackOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult PackOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inner_dims_pos;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'inner_dims_pos'");
    if (namedAttrIt->getName() == getInnerDimsPosAttrName()) {
      tblgen_inner_dims_pos = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_inner_tiles;
  ::mlir::Attribute tblgen_outer_dims_perm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_inner_tiles'");
    if (namedAttrIt->getName() == getStaticInnerTilesAttrName()) {
      tblgen_static_inner_tiles = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOuterDimsPermAttrName()) {
      tblgen_outer_dims_perm = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_outer_dims_perm, "outer_dims_perm")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_inner_dims_pos, "inner_dims_pos")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_inner_tiles, "static_inner_tiles")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches type of dest");
  return ::mlir::success();
}

::mlir::LogicalResult PackOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void PackOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::LogicalResult PackOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult PackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> padding_valueOperands;
  ::llvm::SMLoc padding_valueOperandsLoc;
  (void)padding_valueOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> padding_valueTypes;
  ::mlir::DenseI64ArrayAttr outer_dims_permAttr;
  ::mlir::DenseI64ArrayAttr inner_dims_posAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inner_tilesOperands;
  ::llvm::SMLoc inner_tilesOperandsLoc;
  (void)inner_tilesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_inner_tilesAttr;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("padding_value"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    padding_valueOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      padding_valueOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      padding_valueTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("outer_dims_perm"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(outer_dims_permAttr, ::mlir::Type{}, "outer_dims_perm",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseKeyword("inner_dims_pos"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(inner_dims_posAttr, ::mlir::Type{}, "inner_dims_pos",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("inner_tiles"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  {
    inner_tilesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, inner_tilesOperands, static_inner_tilesAttr))
      return ::mlir::failure();
    result.addAttribute("static_inner_tiles", static_inner_tilesAttr);
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(padding_valueOperands.size()), static_cast<int32_t>(inner_tilesOperands.size())}));
  for (::mlir::Type type : destTypes) {
    (void)type;
    if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'dest' must be ranked tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(destTypes[0]);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(padding_valueOperands, padding_valueTypes, padding_valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inner_tilesOperands, odsBuildableType0, inner_tilesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PackOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  if (getPaddingValue()) {
    _odsPrinter << ' ' << "padding_value";
    _odsPrinter << "(";
    if (::mlir::Value value = getPaddingValue())
      _odsPrinter << value;
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << (getPaddingValue() ? ::llvm::ArrayRef<::mlir::Type>(getPaddingValue().getType()) : ::llvm::ArrayRef<::mlir::Type>());
    _odsPrinter << ")";
  }
  if ((*this)->getAttr("outer_dims_perm")) {
    _odsPrinter << ' ' << "outer_dims_perm";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getOuterDimsPermAttr());
  }
  _odsPrinter << ' ' << "inner_dims_pos";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getInnerDimsPosAttr());
  _odsPrinter << ' ' << "inner_tiles";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getInnerTiles(), getStaticInnerTilesAttr());
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("outer_dims_perm");
  elidedAttrs.push_back("inner_dims_pos");
  elidedAttrs.push_back("static_inner_tiles");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getOuterDimsPermAttr();
     if(attr && (attr == odsBuilder.getDenseI64ArrayAttr({})))
       elidedAttrs.push_back("outer_dims_perm");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void PackOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::PackOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::PadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PadOpGenericAdaptorBase::PadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.pad", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, PadOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr PadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr PadOpGenericAdaptorBase::getStaticLowAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, PadOp::getStaticLowAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> PadOpGenericAdaptorBase::getStaticLow() {
  auto attr = getStaticLowAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr PadOpGenericAdaptorBase::getStaticHighAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, PadOp::getStaticHighAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> PadOpGenericAdaptorBase::getStaticHigh() {
  auto attr = getStaticHighAttr();
  return attr;
}

::mlir::UnitAttr PadOpGenericAdaptorBase::getNofoldAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, PadOp::getNofoldAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool PadOpGenericAdaptorBase::getNofold() {
  auto attr = getNofoldAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &PadOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PadOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
PadOpAdaptor::PadOpAdaptor(PadOp op) : PadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nofold;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.pad' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == PadOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == PadOp::getNofoldAttrName(*odsOpName)) {
      tblgen_nofold = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_high;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.pad' op ""requires attribute 'static_high'");
    if (namedAttrIt->getName() == PadOp::getStaticHighAttrName(*odsOpName)) {
      tblgen_static_high = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_low;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.pad' op ""requires attribute 'static_low'");
    if (namedAttrIt->getName() == PadOp::getStaticLowAttrName(*odsOpName)) {
      tblgen_static_low = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'tensor.pad' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_low && !((tblgen_static_low.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.pad' op ""attribute 'static_low' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_high && !((tblgen_static_high.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.pad' op ""attribute 'static_high' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_nofold && !((tblgen_nofold.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'tensor.pad' op ""attribute 'nofold' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PadOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range PadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> PadOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range PadOp::getLow() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range PadOp::getHigh() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange PadOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PadOp::getLowMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange PadOp::getHighMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> PadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> PadOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::Region &PadOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::DenseI64ArrayAttr PadOp::getStaticLowAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getStaticLowAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> PadOp::getStaticLow() {
  auto attr = getStaticLowAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr PadOp::getStaticHighAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getStaticHighAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> PadOp::getStaticHigh() {
  auto attr = getStaticHighAttr();
  return attr;
}

::mlir::UnitAttr PadOp::getNofoldAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getNofoldAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool PadOp::getNofold() {
  auto attr = getNofoldAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void PadOp::setStaticLowAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticLowAttrName(), attr);
}

void PadOp::setStaticLow(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticLowAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void PadOp::setStaticHighAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticHighAttrName(), attr);
}

void PadOp::setStaticHigh(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticHighAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void PadOp::setNofoldAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNofoldAttrName(), attr);
}

void PadOp::setNofold(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNofoldAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getNofoldAttrName());
}

::mlir::Attribute PadOp::removeNofoldAttr() {
  return (*this)->removeAttr(getNofoldAttrName());
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::mlir::DenseI64ArrayAttr static_low, ::mlir::DenseI64ArrayAttr static_high, /*optional*/::mlir::UnitAttr nofold) {
  odsState.addOperands(source);
  odsState.addOperands(low);
  odsState.addOperands(high);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(low.size()), static_cast<int32_t>(high.size())}));
  odsState.addAttribute(getStaticLowAttrName(odsState.name), static_low);
  odsState.addAttribute(getStaticHighAttrName(odsState.name), static_high);
  if (nofold) {
    odsState.addAttribute(getNofoldAttrName(odsState.name), nofold);
  }
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::mlir::DenseI64ArrayAttr static_low, ::mlir::DenseI64ArrayAttr static_high, /*optional*/::mlir::UnitAttr nofold) {
  odsState.addOperands(source);
  odsState.addOperands(low);
  odsState.addOperands(high);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(low.size()), static_cast<int32_t>(high.size())}));
  odsState.addAttribute(getStaticLowAttrName(odsState.name), static_low);
  odsState.addAttribute(getStaticHighAttrName(odsState.name), static_high);
  if (nofold) {
    odsState.addAttribute(getNofoldAttrName(odsState.name), nofold);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::llvm::ArrayRef<int64_t> static_low, ::llvm::ArrayRef<int64_t> static_high, /*optional*/bool nofold) {
  odsState.addOperands(source);
  odsState.addOperands(low);
  odsState.addOperands(high);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(low.size()), static_cast<int32_t>(high.size())}));
  odsState.addAttribute(getStaticLowAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_low));
  odsState.addAttribute(getStaticHighAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_high));
  if (nofold) {
    odsState.addAttribute(getNofoldAttrName(odsState.name), ((nofold) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::llvm::ArrayRef<int64_t> static_low, ::llvm::ArrayRef<int64_t> static_high, /*optional*/bool nofold) {
  odsState.addOperands(source);
  odsState.addOperands(low);
  odsState.addOperands(high);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(low.size()), static_cast<int32_t>(high.size())}));
  odsState.addAttribute(getStaticLowAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_low));
  odsState.addAttribute(getStaticHighAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_high));
  if (nofold) {
    odsState.addAttribute(getNofoldAttrName(odsState.name), ((nofold) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nofold;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNofoldAttrName()) {
      tblgen_nofold = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_high;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_high'");
    if (namedAttrIt->getName() == getStaticHighAttrName()) {
      tblgen_static_high = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_low;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_low'");
    if (namedAttrIt->getName() == getStaticLowAttrName()) {
      tblgen_static_low = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_low, "static_low")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_high, "static_high")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps2(*this, tblgen_nofold, "nofold")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_TensorOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult PadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lowOperands;
  ::llvm::SMLoc lowOperandsLoc;
  (void)lowOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_lowAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> highOperands;
  ::llvm::SMLoc highOperandsLoc;
  (void)highOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_highAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("nofold"))) {
    result.addAttribute("nofold", parser.getBuilder().getUnitAttr());
  }
  if (parser.parseKeyword("low"))
    return ::mlir::failure();
  {
    lowOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, lowOperands, static_lowAttr))
      return ::mlir::failure();
    result.addAttribute("static_low", static_lowAttr);
  }
  if (parser.parseKeyword("high"))
    return ::mlir::failure();
  {
    highOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, highOperands, static_highAttr))
      return ::mlir::failure();
    result.addAttribute("static_high", static_highAttr);
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();

  ensureTerminator(*regionRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(lowOperands.size()), static_cast<int32_t>(highOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(lowOperands, odsBuildableType0, lowOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(highOperands, odsBuildableType0, highOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  if ((*this)->getAttr("nofold")) {
    _odsPrinter << ' ' << "nofold";
  }
  _odsPrinter << ' ' << "low";
  printDynamicIndexList(_odsPrinter, *this, getLow(), getStaticLowAttr());
  _odsPrinter << ' ' << "high";
  printDynamicIndexList(_odsPrinter, *this, getHigh(), getStaticHighAttr());
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getRegion().empty() ? nullptr : getRegion().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("nofold");
  elidedAttrs.push_back("static_low");
  elidedAttrs.push_back("static_high");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNofoldAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("nofold");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void PadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::PadOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ParallelInsertSliceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ParallelInsertSliceOpGenericAdaptorBase::ParallelInsertSliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.parallel_insert_slice", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ParallelInsertSliceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ParallelInsertSliceOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ParallelInsertSliceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ParallelInsertSliceOpGenericAdaptorBase::getStaticOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, ParallelInsertSliceOp::getStaticOffsetsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ParallelInsertSliceOpGenericAdaptorBase::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ParallelInsertSliceOpGenericAdaptorBase::getStaticSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ParallelInsertSliceOp::getStaticSizesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ParallelInsertSliceOpGenericAdaptorBase::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ParallelInsertSliceOpGenericAdaptorBase::getStaticStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, ParallelInsertSliceOp::getStaticStridesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ParallelInsertSliceOpGenericAdaptorBase::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

} // namespace detail
ParallelInsertSliceOpAdaptor::ParallelInsertSliceOpAdaptor(ParallelInsertSliceOp op) : ParallelInsertSliceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ParallelInsertSliceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.parallel_insert_slice' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ParallelInsertSliceOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.parallel_insert_slice' op ""requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == ParallelInsertSliceOp::getStaticOffsetsAttrName(*odsOpName)) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.parallel_insert_slice' op ""requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == ParallelInsertSliceOp::getStaticSizesAttrName(*odsOpName)) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.parallel_insert_slice' op ""requires attribute 'static_strides'");
    if (namedAttrIt->getName() == ParallelInsertSliceOp::getStaticStridesAttrName(*odsOpName)) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'tensor.parallel_insert_slice' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_offsets && !((tblgen_static_offsets.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.parallel_insert_slice' op ""attribute 'static_offsets' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_sizes && !((tblgen_static_sizes.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.parallel_insert_slice' op ""attribute 'static_sizes' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_strides && !((tblgen_static_strides.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.parallel_insert_slice' op ""attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ParallelInsertSliceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ParallelInsertSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ParallelInsertSliceOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> ParallelInsertSliceOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range ParallelInsertSliceOp::getOffsets() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ParallelInsertSliceOp::getSizes() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ParallelInsertSliceOp::getStrides() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange ParallelInsertSliceOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelInsertSliceOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelInsertSliceOp::getOffsetsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelInsertSliceOp::getSizesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelInsertSliceOp::getStridesMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ParallelInsertSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ParallelInsertSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::DenseI64ArrayAttr ParallelInsertSliceOp::getStaticOffsetsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getStaticOffsetsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ParallelInsertSliceOp::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ParallelInsertSliceOp::getStaticSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getStaticSizesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ParallelInsertSliceOp::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ParallelInsertSliceOp::getStaticStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getStaticStridesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ParallelInsertSliceOp::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

void ParallelInsertSliceOp::setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticOffsetsAttrName(), attr);
}

void ParallelInsertSliceOp::setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticOffsetsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ParallelInsertSliceOp::setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticSizesAttrName(), attr);
}

void ParallelInsertSliceOp::setStaticSizes(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticSizesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ParallelInsertSliceOp::setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticStridesAttrName(), attr);
}

void ParallelInsertSliceOp::setStaticStrides(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticStridesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ParallelInsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
}

void ParallelInsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelInsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
}

void ParallelInsertSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelInsertSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ParallelInsertSliceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == getStaticOffsetsAttrName()) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == getStaticSizesAttrName()) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_strides'");
    if (namedAttrIt->getName() == getStaticStridesAttrName()) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_offsets, "static_offsets")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_sizes, "static_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_strides, "static_strides")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ParallelInsertSliceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ParallelInsertSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_offsetsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_sizesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, offsetsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, stridesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ParallelInsertSliceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  printDynamicIndexList(_odsPrinter, *this, getOffsets(), getStaticOffsetsAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getSizes(), getStaticSizesAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getStrides(), getStaticStridesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("static_offsets");
  elidedAttrs.push_back("static_sizes");
  elidedAttrs.push_back("static_strides");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ParallelInsertSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::RankOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RankOpGenericAdaptorBase::RankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.rank", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RankOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr RankOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
RankOpAdaptor::RankOpAdaptor(RankOp op) : RankOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult RankOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RankOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RankOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> RankOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange RankOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RankOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RankOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(resultType0);
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor) {
  odsState.addOperands(tensor);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(RankOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RankOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(RankOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult RankOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RankOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult RankOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult RankOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RankOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void RankOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::RankOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ReshapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReshapeOpGenericAdaptorBase::ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.reshape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReshapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReshapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp op) : ReshapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReshapeOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> ReshapeOp::getShape() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ReshapeOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::getShapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReshapeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TensorType resultType, Value operand, Value shape) {
       odsState.addOperands(operand);
       odsState.addOperands(shape);
       odsState.addTypes(resultType);
     
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value shape) {
  odsState.addOperands(source);
  odsState.addOperands(shape);
  odsState.addTypes(result);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value shape) {
  odsState.addOperands(source);
  odsState.addOperands(shape);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReshapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReshapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand shapeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> shapeOperands(shapeRawOperands);  ::llvm::SMLoc shapeOperandsLoc;
  (void)shapeOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  shapeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(shapeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(sourceOperands, shapeOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << "(";
  _odsPrinter << getShape();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void ReshapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ReshapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ScatterOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ScatterOpGenericAdaptorBase::ScatterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.scatter", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ScatterOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ScatterOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ScatterOpGenericAdaptorBase::getScatterDimsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ScatterOp::getScatterDimsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ScatterOpGenericAdaptorBase::getScatterDims() {
  auto attr = getScatterDimsAttr();
  return attr;
}

::mlir::UnitAttr ScatterOpGenericAdaptorBase::getUniqueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ScatterOp::getUniqueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool ScatterOpGenericAdaptorBase::getUnique() {
  auto attr = getUniqueAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
ScatterOpAdaptor::ScatterOpAdaptor(ScatterOp op) : ScatterOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ScatterOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_scatter_dims;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.scatter' op ""requires attribute 'scatter_dims'");
    if (namedAttrIt->getName() == ScatterOp::getScatterDimsAttrName(*odsOpName)) {
      tblgen_scatter_dims = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_unique;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ScatterOp::getUniqueAttrName(*odsOpName)) {
      tblgen_unique = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_scatter_dims && !((tblgen_scatter_dims.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.scatter' op ""attribute 'scatter_dims' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_unique && !((tblgen_unique.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'tensor.scatter' op ""attribute 'unique' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ScatterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScatterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ScatterOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> ScatterOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> ScatterOp::getIndices() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(2).begin());
}

::mlir::MutableOperandRange ScatterOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScatterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScatterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ScatterOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr ScatterOp::getScatterDimsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getScatterDimsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ScatterOp::getScatterDims() {
  auto attr = getScatterDimsAttr();
  return attr;
}

::mlir::UnitAttr ScatterOp::getUniqueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getUniqueAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ScatterOp::getUnique() {
  auto attr = getUniqueAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void ScatterOp::setScatterDimsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getScatterDimsAttrName(), attr);
}

void ScatterOp::setScatterDims(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getScatterDimsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ScatterOp::setUniqueAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUniqueAttrName(), attr);
}

void ScatterOp::setUnique(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getUniqueAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getUniqueAttrName());
}

::mlir::Attribute ScatterOp::removeUniqueAttr() {
  return (*this)->removeAttr(getUniqueAttrName());
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr scatter_dims, /*optional*/::mlir::UnitAttr unique) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  odsState.addAttribute(getScatterDimsAttrName(odsState.name), scatter_dims);
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), unique);
  }
  odsState.addTypes(result);
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr scatter_dims, /*optional*/::mlir::UnitAttr unique) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  odsState.addAttribute(getScatterDimsAttrName(odsState.name), scatter_dims);
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), unique);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> scatter_dims, /*optional*/bool unique) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  odsState.addAttribute(getScatterDimsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(scatter_dims));
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), ((unique) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(result);
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> scatter_dims, /*optional*/bool unique) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(indices);
  odsState.addAttribute(getScatterDimsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(scatter_dims));
  if (unique) {
    odsState.addAttribute(getUniqueAttrName(odsState.name), ((unique) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScatterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScatterOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_scatter_dims;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'scatter_dims'");
    if (namedAttrIt->getName() == getScatterDimsAttrName()) {
      tblgen_scatter_dims = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_unique;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getUniqueAttrName()) {
      tblgen_unique = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_scatter_dims, "scatter_dims")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps2(*this, tblgen_unique, "unique")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ScatterOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ScatterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand indicesRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> indicesOperands(indicesRawOperands);  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::DenseI64ArrayAttr scatter_dimsAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indicesRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseKeyword("scatter_dims"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(scatter_dimsAttr, ::mlir::Type{}, "scatter_dims",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("unique"))) {
    result.addAttribute("unique", parser.getBuilder().getUnitAttr());
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(sourceOperands, destOperands, indicesOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScatterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ' ' << "scatter_dims";
  _odsPrinter << "(";
_odsPrinter.printStrippedAttrOrType(getScatterDimsAttr());
  _odsPrinter << ")";
  if ((*this)->getAttr("unique")) {
    _odsPrinter << ' ' << "unique";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("scatter_dims");
  elidedAttrs.push_back("unique");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getUniqueAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("unique");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void ScatterOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::ScatterOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::SplatOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SplatOpGenericAdaptorBase::SplatOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.splat", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SplatOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SplatOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
SplatOpAdaptor::SplatOpAdaptor(SplatOp op) : SplatOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SplatOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SplatOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SplatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOp::getInput() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SplatOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SplatOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SplatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> SplatOp::getAggregate() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value element, Type aggregateType) {
 build(odsBuilder, odsState, aggregateType, element); 
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type aggregate, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(aggregate);
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplatOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SplatOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSResults(0).begin()).getType().cast<TensorType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that operand type matches element type of result");
  return ::mlir::success();
}

::mlir::LogicalResult SplatOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SplatOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type aggregateRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> aggregateTypes(aggregateRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aggregateRawTypes[0] = type;
  }
  for (::mlir::Type type : aggregateTypes) {
    (void)type;
    if (!((((type.isa<::mlir::RankedTensorType>())) && ((type.cast<::mlir::ShapedType>().hasStaticShape()))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'aggregate' must be statically shaped tensor of any type values, but got " << type;
    }
  }
  result.addTypes(aggregateTypes);
  if (parser.resolveOperands(inputOperands, aggregateTypes[0].cast<TensorType>().getElementType(), inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SplatOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAggregate().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SplatOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::SplatOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::UnPackOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UnPackOpGenericAdaptorBase::UnPackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.unpack", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UnPackOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr UnPackOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr UnPackOpGenericAdaptorBase::getOuterDimsPermAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, UnPackOp::getOuterDimsPermAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseI64ArrayAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getDenseI64ArrayAttr({});
  return attr;
}

::llvm::ArrayRef<int64_t> UnPackOpGenericAdaptorBase::getOuterDimsPerm() {
  auto attr = getOuterDimsPermAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getDenseI64ArrayAttr({});
  return attr;
}

::mlir::DenseI64ArrayAttr UnPackOpGenericAdaptorBase::getInnerDimsPosAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, UnPackOp::getInnerDimsPosAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> UnPackOpGenericAdaptorBase::getInnerDimsPos() {
  auto attr = getInnerDimsPosAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr UnPackOpGenericAdaptorBase::getStaticInnerTilesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, UnPackOp::getStaticInnerTilesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> UnPackOpGenericAdaptorBase::getStaticInnerTiles() {
  auto attr = getStaticInnerTilesAttr();
  return attr;
}

} // namespace detail
UnPackOpAdaptor::UnPackOpAdaptor(UnPackOp op) : UnPackOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UnPackOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inner_dims_pos;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.unpack' op ""requires attribute 'inner_dims_pos'");
    if (namedAttrIt->getName() == UnPackOp::getInnerDimsPosAttrName(*odsOpName)) {
      tblgen_inner_dims_pos = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_inner_tiles;
  ::mlir::Attribute tblgen_outer_dims_perm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tensor.unpack' op ""requires attribute 'static_inner_tiles'");
    if (namedAttrIt->getName() == UnPackOp::getStaticInnerTilesAttrName(*odsOpName)) {
      tblgen_static_inner_tiles = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == UnPackOp::getOuterDimsPermAttrName(*odsOpName)) {
      tblgen_outer_dims_perm = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_outer_dims_perm && !((tblgen_outer_dims_perm.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.unpack' op ""attribute 'outer_dims_perm' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_inner_dims_pos && !((tblgen_inner_dims_pos.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.unpack' op ""attribute 'inner_dims_pos' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_inner_tiles && !((tblgen_static_inner_tiles.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'tensor.unpack' op ""attribute 'static_inner_tiles' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UnPackOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range UnPackOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> UnPackOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> UnPackOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range UnPackOp::getInnerTiles() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange UnPackOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UnPackOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UnPackOp::getInnerTilesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UnPackOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnPackOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> UnPackOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr UnPackOp::getOuterDimsPermAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOuterDimsPermAttrName()).dyn_cast_or_null<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> UnPackOp::getOuterDimsPerm() {
  auto attr = getOuterDimsPermAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr({});
  return attr;
}

::mlir::DenseI64ArrayAttr UnPackOp::getInnerDimsPosAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getInnerDimsPosAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> UnPackOp::getInnerDimsPos() {
  auto attr = getInnerDimsPosAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr UnPackOp::getStaticInnerTilesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStaticInnerTilesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> UnPackOp::getStaticInnerTiles() {
  auto attr = getStaticInnerTilesAttr();
  return attr;
}

void UnPackOp::setOuterDimsPermAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getOuterDimsPermAttrName(), attr);
}

void UnPackOp::setOuterDimsPerm(::std::optional<::llvm::ArrayRef<int64_t>> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOuterDimsPermAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(*attrValue));
    (*this)->removeAttr(getOuterDimsPermAttrName());
}

void UnPackOp::setInnerDimsPosAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getInnerDimsPosAttrName(), attr);
}

void UnPackOp::setInnerDimsPos(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getInnerDimsPosAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void UnPackOp::setStaticInnerTilesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticInnerTilesAttrName(), attr);
}

void UnPackOp::setStaticInnerTiles(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticInnerTilesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

::mlir::Attribute UnPackOp::removeOuterDimsPermAttr() {
  return (*this)->removeAttr(getOuterDimsPermAttrName());
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(inner_tiles);
  if (outer_dims_perm) {
    odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), outer_dims_perm);
  }
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), inner_dims_pos);
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), static_inner_tiles);
  odsState.addTypes(result);
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(inner_tiles);
  if (outer_dims_perm) {
    odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), outer_dims_perm);
  }
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), inner_dims_pos);
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), static_inner_tiles);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(UnPackOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(inner_tiles);
  if (outer_dims_perm) {
    odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), outer_dims_perm);
  }
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), inner_dims_pos);
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), static_inner_tiles);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(outer_dims_perm));
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(inner_dims_pos));
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_inner_tiles));
  odsState.addTypes(result);
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(outer_dims_perm));
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(inner_dims_pos));
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_inner_tiles));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(UnPackOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addOperands(inner_tiles);
  odsState.addAttribute(getOuterDimsPermAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(outer_dims_perm));
  odsState.addAttribute(getInnerDimsPosAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(inner_dims_pos));
  odsState.addAttribute(getStaticInnerTilesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_inner_tiles));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnPackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnPackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(UnPackOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult UnPackOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inner_dims_pos;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'inner_dims_pos'");
    if (namedAttrIt->getName() == getInnerDimsPosAttrName()) {
      tblgen_inner_dims_pos = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_inner_tiles;
  ::mlir::Attribute tblgen_outer_dims_perm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_inner_tiles'");
    if (namedAttrIt->getName() == getStaticInnerTilesAttrName()) {
      tblgen_static_inner_tiles = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOuterDimsPermAttrName()) {
      tblgen_outer_dims_perm = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_outer_dims_perm, "outer_dims_perm")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_inner_dims_pos, "inner_dims_pos")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_TensorOps1(*this, tblgen_static_inner_tiles, "static_inner_tiles")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches type of dest");
  return ::mlir::success();
}

::mlir::LogicalResult UnPackOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void UnPackOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::LogicalResult UnPackOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult UnPackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::DenseI64ArrayAttr outer_dims_permAttr;
  ::mlir::DenseI64ArrayAttr inner_dims_posAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inner_tilesOperands;
  ::llvm::SMLoc inner_tilesOperandsLoc;
  (void)inner_tilesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_inner_tilesAttr;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("outer_dims_perm"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(outer_dims_permAttr, ::mlir::Type{}, "outer_dims_perm",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseKeyword("inner_dims_pos"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(inner_dims_posAttr, ::mlir::Type{}, "inner_dims_pos",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("inner_tiles"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  {
    inner_tilesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, inner_tilesOperands, static_inner_tilesAttr))
      return ::mlir::failure();
    result.addAttribute("static_inner_tiles", static_inner_tilesAttr);
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  for (::mlir::Type type : destTypes) {
    (void)type;
    if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'dest' must be ranked tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(destTypes[0]);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inner_tilesOperands, odsBuildableType0, inner_tilesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UnPackOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  if ((*this)->getAttr("outer_dims_perm")) {
    _odsPrinter << ' ' << "outer_dims_perm";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
  _odsPrinter.printStrippedAttrOrType(getOuterDimsPermAttr());
  }
  _odsPrinter << ' ' << "inner_dims_pos";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getInnerDimsPosAttr());
  _odsPrinter << ' ' << "inner_tiles";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getInnerTiles(), getStaticInnerTilesAttr());
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("outer_dims_perm");
  elidedAttrs.push_back("inner_dims_pos");
  elidedAttrs.push_back("static_inner_tiles");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getOuterDimsPermAttr();
     if(attr && (attr == odsBuilder.getDenseI64ArrayAttr({})))
       elidedAttrs.push_back("outer_dims_perm");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UnPackOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::UnPackOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::YieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
YieldOpGenericAdaptorBase::YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tensor.yield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> YieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr YieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : YieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value YieldOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange YieldOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 /* nothing to do */ 
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_TensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tensor::YieldOp)


#endif  // GET_OP_CLASSES

