/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl::ApplyNativeConstraintOp,
::mlir::pdl::ApplyNativeRewriteOp,
::mlir::pdl::AttributeOp,
::mlir::pdl::EraseOp,
::mlir::pdl::OperandOp,
::mlir::pdl::OperandsOp,
::mlir::pdl::OperationOp,
::mlir::pdl::PatternOp,
::mlir::pdl::RangeOp,
::mlir::pdl::ReplaceOp,
::mlir::pdl::ResultOp,
::mlir::pdl::ResultsOp,
::mlir::pdl::RewriteOp,
::mlir::pdl::TypeOp,
::mlir::pdl::TypesOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::PDLType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::TypeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::AttributeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::ValueType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle for an `mlir::Value` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::ValueType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && (((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())) || ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` or PDL handle for an `mlir::Value` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!attr.cast<::mlir::IntegerAttr>().getValue().isNegative())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_PDLOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_PDLOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace pdl
} // namespace mlir
namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ApplyNativeConstraintOp definitions
//===----------------------------------------------------------------------===//

ApplyNativeConstraintOpAdaptor::ApplyNativeConstraintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.apply_native_constraint", odsAttrs.getContext());
}

ApplyNativeConstraintOpAdaptor::ApplyNativeConstraintOpAdaptor(ApplyNativeConstraintOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ApplyNativeConstraintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ApplyNativeConstraintOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyNativeConstraintOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyNativeConstraintOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyNativeConstraintOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyNativeConstraintOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.apply_native_constraint' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == ApplyNativeConstraintOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.apply_native_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyNativeConstraintOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyNativeConstraintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyNativeConstraintOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyNativeConstraintOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyNativeConstraintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyNativeConstraintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr ApplyNativeConstraintOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyNativeConstraintOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyNativeConstraintOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyNativeConstraintOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyNativeConstraintOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyNativeConstraintOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ApplyNativeConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyNativeConstraintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "(";
  _odsPrinter << getArgs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ApplyNativeConstraintOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ApplyNativeRewriteOp definitions
//===----------------------------------------------------------------------===//

ApplyNativeRewriteOpAdaptor::ApplyNativeRewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.apply_native_rewrite", odsAttrs.getContext());
}

ApplyNativeRewriteOpAdaptor::ApplyNativeRewriteOpAdaptor(ApplyNativeRewriteOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ApplyNativeRewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ApplyNativeRewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyNativeRewriteOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyNativeRewriteOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyNativeRewriteOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyNativeRewriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.apply_native_rewrite' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == ApplyNativeRewriteOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.apply_native_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyNativeRewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyNativeRewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyNativeRewriteOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyNativeRewriteOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyNativeRewriteOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ApplyNativeRewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ApplyNativeRewriteOp::getResults() {
  return getODSResults(0);
}

::mlir::StringAttr ApplyNativeRewriteOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyNativeRewriteOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyNativeRewriteOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyNativeRewriteOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(results);
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addTypes(results);
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyNativeRewriteOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyNativeRewriteOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ApplyNativeRewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyNativeRewriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  if (!getArgs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
    _odsPrinter << ")";
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ApplyNativeRewriteOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::AttributeOp definitions
//===----------------------------------------------------------------------===//

AttributeOpAdaptor::AttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.attribute", odsAttrs.getContext());
}

AttributeOpAdaptor::AttributeOpAdaptor(AttributeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> AttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange AttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOpAdaptor::getValueType() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange AttributeOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr AttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute AttributeOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AttributeOp::getValueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::Attribute>();
  return attr;
}

::std::optional<::mlir::Attribute> AttributeOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr ? ::std::optional<::mlir::Attribute>(attr) : (::std::nullopt);
}

::mlir::LogicalResult AttributeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AttributeOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((true)))
    return emitError(loc, "'pdl.attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AttributeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AttributeOp::getValueType() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange AttributeOp::getValueTypeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::AttributeType> AttributeOp::getAttr() {
  return *getODSResults(0).begin();
}

::mlir::Attribute AttributeOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).dyn_cast_or_null<::mlir::Attribute>();
}

::std::optional<::mlir::Attribute> AttributeOp::getValue() {
  auto attr = getValueAttr();
  return attr ? ::std::optional<::mlir::Attribute>(attr) : (::std::nullopt);
}

void AttributeOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

::mlir::Attribute AttributeOp::removeValueAttr() {
  return (*this)->removeAttr(getValueAttrName());
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value type) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), type,
            Attribute());
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute attr) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), Value(), attr);
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attr, /*optional*/::mlir::Value valueType, /*optional*/::mlir::Attribute value) {
  if (valueType)
    odsState.addOperands(valueType);
  if (value) {
    odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  odsState.addTypes(attr);
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value valueType, /*optional*/::mlir::Attribute value) {
  if (valueType)
    odsState.addOperands(valueType);
  if (value) {
    odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AttributeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps1(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AttributeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> valueTypeOperands;
  ::llvm::SMLoc valueTypeOperandsLoc;
  (void)valueTypeOperandsLoc;
  ::mlir::Attribute valueAttr;
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  {
    valueTypeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      valueTypeOperands.push_back(operand);
    }
  }
  }
  if (::mlir::succeeded(parser.parseOptionalEqual())) {

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}, "value", result.attributes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueTypeOperands, odsBuildableType1, valueTypeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getValueType()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    if (::mlir::Value value = getValueType())
      _odsPrinter << value;
  }
  if ((*this)->getAttr("value")) {
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printAttribute(getValueAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::AttributeOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::EraseOp definitions
//===----------------------------------------------------------------------===//

EraseOpAdaptor::EraseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.erase", odsAttrs.getContext());
}

EraseOpAdaptor::EraseOpAdaptor(EraseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> EraseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EraseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EraseOpAdaptor::getOpValue() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange EraseOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr EraseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult EraseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EraseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EraseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> EraseOp::getOpValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EraseOp::getOpValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> EraseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EraseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value opValue) {
  odsState.addOperands(opValue);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opValue) {
  odsState.addOperands(opValue);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EraseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EraseOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EraseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EraseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand opValueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> opValueOperands(opValueRawOperands);  ::llvm::SMLoc opValueOperandsLoc;
  (void)opValueOperandsLoc;

  opValueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opValueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(opValueOperands, odsBuildableType0, opValueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOpValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::EraseOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperandOp definitions
//===----------------------------------------------------------------------===//

OperandOpAdaptor::OperandOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.operand", odsAttrs.getContext());
}

OperandOpAdaptor::OperandOpAdaptor(OperandOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> OperandOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange OperandOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandOpAdaptor::getValueType() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange OperandOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr OperandOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OperandOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OperandOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OperandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandOp::getValueType() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange OperandOp::getValueTypeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OperandOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OperandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::ValueType> OperandOp::getValue() {
  return *getODSResults(0).begin();
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, odsBuilder.getType<ValueType>(), Value());
    
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, /*optional*/::mlir::Value valueType) {
  if (valueType)
    odsState.addOperands(valueType);
  odsState.addTypes(value);
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value valueType) {
  if (valueType)
    odsState.addOperands(valueType);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OperandOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OperandOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OperandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> valueTypeOperands;
  ::llvm::SMLoc valueTypeOperandsLoc;
  (void)valueTypeOperandsLoc;
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  {
    valueTypeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      valueTypeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueTypeOperands, odsBuildableType1, valueTypeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getValueType()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    if (::mlir::Value value = getValueType())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::OperandOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperandsOp definitions
//===----------------------------------------------------------------------===//

OperandsOpAdaptor::OperandsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.operands", odsAttrs.getContext());
}

OperandsOpAdaptor::OperandsOpAdaptor(OperandsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> OperandsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange OperandsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OperandsOpAdaptor::getValueType() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange OperandsOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr OperandsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OperandsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OperandsOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OperandsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> OperandsOp::getValueType() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange OperandsOp::getValueTypeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OperandsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OperandsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> OperandsOp::getValue() {
  return *getODSResults(0).begin();
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, RangeType::get(odsBuilder.getType<ValueType>()),
            Value());
    
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, /*optional*/::mlir::Value valueType) {
  if (valueType)
    odsState.addOperands(valueType);
  odsState.addTypes(value);
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value valueType) {
  if (valueType)
    odsState.addOperands(valueType);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperandsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OperandsOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OperandsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OperandsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> valueTypeOperands;
  ::llvm::SMLoc valueTypeOperandsLoc;
  (void)valueTypeOperandsLoc;
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  {
    valueTypeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      valueTypeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::ValueType>());
  ::mlir::Type odsBuildableType1 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueTypeOperands, odsBuildableType1, valueTypeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperandsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getValueType()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    if (::mlir::Value value = getValueType())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::OperandsOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::OperationOp definitions
//===----------------------------------------------------------------------===//

OperationOpAdaptor::OperationOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.operation", odsAttrs.getContext());
}

OperationOpAdaptor::OperationOpAdaptor(OperationOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> OperationOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, OperationOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange OperationOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange OperationOpAdaptor::getOperandValues() {
  return getODSOperands(0);
}

::mlir::ValueRange OperationOpAdaptor::getAttributeValues() {
  return getODSOperands(1);
}

::mlir::ValueRange OperationOpAdaptor::getTypeValues() {
  return getODSOperands(2);
}

::mlir::ValueRange OperationOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr OperationOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr OperationOpAdaptor::getOpNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, OperationOp::getOpNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > OperationOpAdaptor::getOpName() {
  auto attr = getOpNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr OperationOpAdaptor::getAttributeValueNamesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, OperationOp::getAttributeValueNamesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr OperationOpAdaptor::getAttributeValueNames() {
  auto attr = getAttributeValueNamesAttr();
  return attr;
}

::mlir::LogicalResult OperationOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_attributeValueNames;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.operation' op ""requires attribute 'attributeValueNames'");
    if (namedAttrIt->getName() == OperationOp::getAttributeValueNamesAttrName(*odsOpName)) {
      tblgen_attributeValueNames = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_opName;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.operation' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == OperationOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == OperationOp::getOpNameAttrName(*odsOpName)) {
      tblgen_opName = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'pdl.operation' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_opName && !((tblgen_opName.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.operation' op ""attribute 'opName' failed to satisfy constraint: string attribute");

  if (tblgen_attributeValueNames && !(((tblgen_attributeValueNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_attributeValueNames.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'pdl.operation' op ""attribute 'attributeValueNames' failed to satisfy constraint: string array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range OperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OperationOp::getOperandValues() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range OperationOp::getAttributeValues() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range OperationOp::getTypeValues() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange OperationOp::getOperandValuesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange OperationOp::getAttributeValuesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange OperationOp::getTypeValuesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> OperationOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OperationOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> OperationOp::getOp() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr OperationOp::getOpNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOpNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > OperationOp::getOpName() {
  auto attr = getOpNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr OperationOp::getAttributeValueNamesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAttributeValueNamesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr OperationOp::getAttributeValueNames() {
  auto attr = getAttributeValueNamesAttr();
  return attr;
}

void OperationOp::setOpNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getOpNameAttrName(), attr);
}

void OperationOp::setOpName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOpNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getOpNameAttrName());
}

void OperationOp::setAttributeValueNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAttributeValueNamesAttrName(), attr);
}

::mlir::Attribute OperationOp::removeOpNameAttr() {
  return (*this)->removeAttr(getOpNameAttrName());
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Optional<StringRef> name, ValueRange operandValues, ArrayRef<StringRef> attrNames, ValueRange attrValues, ValueRange resultTypes) {
      auto nameAttr = name ? odsBuilder.getStringAttr(*name) : StringAttr();
      build(odsBuilder, odsState, odsBuilder.getType<OperationType>(), nameAttr,
            operandValues, attrValues, odsBuilder.getStrArrayAttr(attrNames),
            resultTypes);
    
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type op, /*optional*/::mlir::StringAttr opName, ::mlir::ValueRange operandValues, ::mlir::ValueRange attributeValues, ::mlir::ArrayAttr attributeValueNames, ::mlir::ValueRange typeValues) {
  odsState.addOperands(operandValues);
  odsState.addOperands(attributeValues);
  odsState.addOperands(typeValues);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(operandValues.size()), static_cast<int32_t>(attributeValues.size()), static_cast<int32_t>(typeValues.size())}));
  if (opName) {
    odsState.addAttribute(getOpNameAttrName(odsState.name), opName);
  }
  odsState.addAttribute(getAttributeValueNamesAttrName(odsState.name), attributeValueNames);
  odsState.addTypes(op);
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr opName, ::mlir::ValueRange operandValues, ::mlir::ValueRange attributeValues, ::mlir::ArrayAttr attributeValueNames, ::mlir::ValueRange typeValues) {
  odsState.addOperands(operandValues);
  odsState.addOperands(attributeValues);
  odsState.addOperands(typeValues);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(operandValues.size()), static_cast<int32_t>(attributeValues.size()), static_cast<int32_t>(typeValues.size())}));
  if (opName) {
    odsState.addAttribute(getOpNameAttrName(odsState.name), opName);
  }
  odsState.addAttribute(getAttributeValueNamesAttrName(odsState.name), attributeValueNames);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OperationOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_attributeValueNames;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'attributeValueNames'");
    if (namedAttrIt->getName() == getAttributeValueNamesAttrName()) {
      tblgen_attributeValueNames = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_opName;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOpNameAttrName()) {
      tblgen_opName = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps0(*this, tblgen_opName, "opName")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps2(*this, tblgen_attributeValueNames, "attributeValueNames")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OperationOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr opNameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandValuesOperands;
  ::llvm::SMLoc operandValuesOperandsLoc;
  (void)operandValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandValuesTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attributeValuesOperands;
  ::llvm::SMLoc attributeValuesOperandsLoc;
  (void)attributeValuesOperandsLoc;
  ::mlir::ArrayAttr attributeValueNamesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> typeValuesOperands;
  ::llvm::SMLoc typeValuesOperandsLoc;
  (void)typeValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> typeValuesTypes;

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(opNameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "opName", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (opNameAttr) {
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  operandValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandValuesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandValuesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    attributeValuesOperandsLoc = parser.getCurrentLocation();
    if (parseOperationOpAttributes(parser, attributeValuesOperands, attributeValueNamesAttr))
      return ::mlir::failure();
    result.addAttribute("attributeValueNames", attributeValueNamesAttr);
  }
  if (::mlir::succeeded(parser.parseOptionalArrow())) {
  if (parser.parseLParen())
    return ::mlir::failure();

  typeValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(typeValuesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(typeValuesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(operandValuesOperands.size()), static_cast<int32_t>(attributeValuesOperands.size()), static_cast<int32_t>(typeValuesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(operandValuesOperands, operandValuesTypes, operandValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attributeValuesOperands, odsBuildableType1, attributeValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(typeValuesOperands, typeValuesTypes, typeValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperationOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("opName")) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getOpNameAttr());
  }
  if (!getOperandValues().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getOperandValues();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getOperandValues().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  printOperationOpAttributes(_odsPrinter, *this, getAttributeValues(), getAttributeValueNamesAttr());
  if (!getTypeValues().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ' << "(";
    _odsPrinter << getTypeValues();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getTypeValues().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("opName");
  elidedAttrs.push_back("attributeValueNames");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::OperationOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::PatternOp definitions
//===----------------------------------------------------------------------===//

PatternOpAdaptor::PatternOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.pattern", odsAttrs.getContext());
}

PatternOpAdaptor::PatternOpAdaptor(PatternOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> PatternOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PatternOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange PatternOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr PatternOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr PatternOpAdaptor::getBenefitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PatternOp::getBenefitAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint16_t PatternOpAdaptor::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr PatternOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PatternOp::getSymNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > PatternOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &PatternOpAdaptor::getBodyRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange PatternOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::LogicalResult PatternOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_benefit;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.pattern' op ""requires attribute 'benefit'");
    if (namedAttrIt->getName() == PatternOp::getBenefitAttrName(*odsOpName)) {
      tblgen_benefit = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PatternOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_benefit && !((((tblgen_benefit.isa<::mlir::IntegerAttr>())) && ((tblgen_benefit.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!tblgen_benefit.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl.pattern' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.pattern' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PatternOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PatternOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> PatternOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PatternOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &PatternOp::getBodyRegion() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr PatternOp::getBenefitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBenefitAttrName()).cast<::mlir::IntegerAttr>();
}

uint16_t PatternOp::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr PatternOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > PatternOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void PatternOp::setBenefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getBenefitAttrName(), attr);
}

void PatternOp::setBenefit(uint16_t attrValue) {
  (*this)->setAttr(getBenefitAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(16), attrValue));
}

void PatternOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void PatternOp::setSymName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSymNameAttrName());
}

::mlir::Attribute PatternOp::removeSymNameAttr() {
  return (*this)->removeAttr(getSymNameAttrName());
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(getBenefitAttrName(odsState.name), benefit);
  if (sym_name) {
    odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(getBenefitAttrName(odsState.name), benefit);
  if (sym_name) {
    odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(getBenefitAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
    odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(getBenefitAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
    odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PatternOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_benefit;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'benefit'");
    if (namedAttrIt->getName() == getBenefitAttrName()) {
      tblgen_benefit = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps3(*this, tblgen_benefit, "benefit")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLOps0(*this, region, "bodyRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult PatternOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PatternOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::IntegerAttr benefitAttr;
  std::unique_ptr<::mlir::Region> bodyRegionRegion = std::make_unique<::mlir::Region>();

  // Parsing an optional symbol name doesn't fail, so no need to check the
  // result.
  (void)parser.parseOptionalSymbolName(sym_nameAttr, "sym_name", result.attributes);
  if (sym_nameAttr) {
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseKeyword("benefit"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(benefitAttr, parser.getBuilder().getIntegerType(16), "benefit",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegionRegion))
    return ::mlir::failure();

  if (bodyRegionRegion->empty()) bodyRegionRegion->emplaceBlock();
  result.addRegion(std::move(bodyRegionRegion));
  return ::mlir::success();
}

void PatternOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("sym_name")) {
    _odsPrinter << ' ';
    _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "benefit";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getBenefitAttr());
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_name");
  elidedAttrs.push_back("benefit");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::PatternOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::RangeOp definitions
//===----------------------------------------------------------------------===//

RangeOpAdaptor::RangeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.range", odsAttrs.getContext());
}

RangeOpAdaptor::RangeOpAdaptor(RangeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> RangeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange RangeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange RangeOpAdaptor::getArguments() {
  return getODSOperands(0);
}

::mlir::ValueRange RangeOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr RangeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult RangeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RangeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range RangeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range RangeOp::getArguments() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange RangeOp::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RangeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RangeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> RangeOp::getResult() {
  return *getODSResults(0).begin();
}

void RangeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addTypes(result);
}

void RangeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RangeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RangeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RangeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argumentsTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (!argumentsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argumentsTypes))
    return ::mlir::failure();
  }
  {
    if (parseRangeType(parser, argumentsTypes, resultRawTypes[0]))
      return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RangeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getArguments().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getArguments();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArguments().getTypes();
  }
  _odsPrinter << ' ';
  printRangeType(_odsPrinter, *this, getArguments().getTypes(), getResult().getType());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void RangeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::RangeOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ReplaceOp definitions
//===----------------------------------------------------------------------===//

ReplaceOpAdaptor::ReplaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.replace", odsAttrs.getContext());
}

ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ReplaceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReplaceOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ReplaceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOpAdaptor::getOpValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReplaceOpAdaptor::getReplOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ReplaceOpAdaptor::getReplValues() {
  return getODSOperands(2);
}

::mlir::ValueRange ReplaceOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ReplaceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.replace' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ReplaceOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'pdl.replace' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ReplaceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> ReplaceOp::getOpValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReplaceOp::getReplOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ReplaceOp::getReplValues() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ReplaceOp::getOpValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReplaceOp::getReplOperationMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReplaceOp::getReplValuesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ReplaceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReplaceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value opValue, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(opValue);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value opValue, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(opValue);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReplaceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReplaceOp::verifyInvariantsImpl() {
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

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReplaceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand opValueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> opValueOperands(opValueRawOperands);  ::llvm::SMLoc opValueOperandsLoc;
  (void)opValueOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> replValuesTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replOperationOperands;
  ::llvm::SMLoc replOperationOperandsLoc;
  (void)replOperationOperandsLoc;

  opValueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(opValueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(replValuesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  {
    replOperationOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      replOperationOperands.push_back(operand);
    }
  }
  if (!replOperationOperands.empty()) {
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(replOperationOperands.size()), static_cast<int32_t>(replValuesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(opValueOperands, odsBuildableType0, opValueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replOperationOperands, odsBuildableType0, replOperationOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOpValue();
  _odsPrinter << ' ' << "with";
  if (!getReplValues().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getReplValues();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getReplValues().getTypes();
    _odsPrinter << ")";
  }
  if (getReplOperation()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getReplOperation())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ReplaceOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ResultOp definitions
//===----------------------------------------------------------------------===//

ResultOpAdaptor::ResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.result", odsAttrs.getContext());
}

ResultOpAdaptor::ResultOpAdaptor(ResultOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultOpAdaptor::getParent() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ResultOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ResultOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ResultOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ResultOp::getIndexAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t ResultOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ResultOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.result' op ""requires attribute 'index'");
    if (namedAttrIt->getName() == ResultOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !(((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl.result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ResultOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> ResultOp::getParent() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResultOp::getParentMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::ValueType> ResultOp::getVal() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ResultOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t ResultOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void ResultOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void ResultOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, ::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  odsState.addTypes(val);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, ::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, uint32_t index) {
  odsState.addOperands(parent);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(val);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, uint32_t index) {
  odsState.addOperands(parent);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResultOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'index'");
    if (namedAttrIt->getName() == getIndexAttrName()) {
      tblgen_index = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ResultOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand parentRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> parentOperands(parentRawOperands);  ::llvm::SMLoc parentOperandsLoc;
  (void)parentOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32), "index",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  parentOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(parentRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(parentOperands, odsBuildableType1, parentOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getParent();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ResultOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ResultOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ResultsOp definitions
//===----------------------------------------------------------------------===//

ResultsOpAdaptor::ResultsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.results", odsAttrs.getContext());
}

ResultsOpAdaptor::ResultsOpAdaptor(ResultsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> ResultsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResultsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultsOpAdaptor::getParent() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ResultsOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr ResultsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ResultsOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ResultsOp::getIndexAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint32_t> ResultsOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

::mlir::LogicalResult ResultsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ResultsOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !(((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl.results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ResultsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResultsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> ResultsOp::getParent() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResultsOp::getParentMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ResultsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResultsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> ResultsOp::getVal() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ResultsOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint32_t> ResultsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void ResultsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void ResultsOp::setIndex(::std::optional<uint32_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue));
    (*this)->removeAttr(getIndexAttrName());
}

::mlir::Attribute ResultsOp::removeIndexAttr() {
  return (*this)->removeAttr(getIndexAttrName());
}

void ResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  if (index) {
    odsState.addAttribute(getIndexAttrName(odsState.name), index);
  }
  odsState.addTypes(val);
}

void ResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  if (index) {
    odsState.addAttribute(getIndexAttrName(odsState.name), index);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResultsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getIndexAttrName()) {
      tblgen_index = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps4(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ResultsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ResultsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand parentRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> parentOperands(parentRawOperands);  ::llvm::SMLoc parentOperandsLoc;
  (void)parentOperandsLoc;
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32), "index", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  parentOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(parentRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseResultsValueType(parser, indexAttr, valRawTypes[0]))
      return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valTypes);
  if (parser.resolveOperands(parentOperands, odsBuildableType0, parentOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResultsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("index")) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getParent();
  _odsPrinter << ' ';
  printResultsValueType(_odsPrinter, *this, getIndexAttr(), getVal().getType());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ResultsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::ResultsOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::RewriteOp definitions
//===----------------------------------------------------------------------===//

RewriteOpAdaptor::RewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.rewrite", odsAttrs.getContext());
}

RewriteOpAdaptor::RewriteOpAdaptor(RewriteOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> RewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, RewriteOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange RewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOpAdaptor::getRoot() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange RewriteOpAdaptor::getExternalArgs() {
  return getODSOperands(1);
}

::mlir::ValueRange RewriteOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr RewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr RewriteOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RewriteOp::getNameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > RewriteOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &RewriteOpAdaptor::getBodyRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange RewriteOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::LogicalResult RewriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.rewrite' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RewriteOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RewriteOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'pdl.rewrite' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RewriteOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range RewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOp::getRoot() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range RewriteOp::getExternalArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RewriteOp::getRootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RewriteOp::getExternalArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RewriteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &RewriteOp::getBodyRegion() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr RewriteOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > RewriteOp::getName() {
  auto attr = getNameAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

void RewriteOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void RewriteOp::setName(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getNameAttrName());
}

::mlir::Attribute RewriteOp::removeNameAttr() {
  return (*this)->removeAttr(getNameAttrName());
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs) {
  if (root)
    odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(root ? 1 : 0), static_cast<int32_t>(externalArgs.size())}));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs) {
  if (root)
    odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(root ? 1 : 0), static_cast<int32_t>(externalArgs.size())}));
  if (name) {
    odsState.addAttribute(getNameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RewriteOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLOps1(*this, region, "bodyRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult RewriteOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult RewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> rootOperands;
  ::llvm::SMLoc rootOperandsLoc;
  (void)rootOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> externalArgsOperands;
  ::llvm::SMLoc externalArgsOperandsLoc;
  (void)externalArgsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> externalArgsTypes;
  std::unique_ptr<::mlir::Region> bodyRegionRegion = std::make_unique<::mlir::Region>();

  {
    rootOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      rootOperands.push_back(operand);
    }
  }
  if (!rootOperands.empty()) {
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("with"))) {

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  externalArgsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(externalArgsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(externalArgsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  }

  {
     auto parseResult = parser.parseOptionalRegion(*bodyRegionRegion);
     if (parseResult.has_value() && failed(*parseResult))
       return ::mlir::failure();
  }
  if (!bodyRegionRegion->empty()) {

  if (bodyRegionRegion->empty()) bodyRegionRegion->emplaceBlock();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(rootOperands.size()), static_cast<int32_t>(externalArgsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(rootOperands, odsBuildableType0, rootOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(externalArgsOperands, externalArgsTypes, externalArgsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RewriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getRoot()) {
    _odsPrinter << ' ';
    if (::mlir::Value value = getRoot())
      _odsPrinter << value;
  }
  if ((*this)->getAttr("name")) {
    _odsPrinter << ' ' << "with";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getNameAttr());
    if (!getExternalArgs().empty()) {
      _odsPrinter << "(";
      _odsPrinter << getExternalArgs();
      _odsPrinter << ' ' << ":";
      _odsPrinter << ' ';
      _odsPrinter << getExternalArgs().getTypes();
      _odsPrinter << ")";
    }
  }
  if (!getBodyRegion().empty()) {
    _odsPrinter << ' ';
    _odsPrinter.printRegion(getBodyRegion());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::RewriteOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::TypeOp definitions
//===----------------------------------------------------------------------===//

TypeOpAdaptor::TypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.type", odsAttrs.getContext());
}

TypeOpAdaptor::TypeOpAdaptor(TypeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange TypeOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr TypeOpAdaptor::getConstantTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TypeOp::getConstantTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::std::optional<::mlir::Type> TypeOpAdaptor::getConstantType() {
  auto attr = getConstantTypeAttr();
  return attr ? ::std::optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::std::nullopt);
}

::mlir::LogicalResult TypeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_constantType;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TypeOp::getConstantTypeAttrName(*odsOpName)) {
      tblgen_constantType = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_constantType && !(((tblgen_constantType.isa<::mlir::TypeAttr>())) && ((tblgen_constantType.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'pdl.type' op ""attribute 'constantType' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::TypeType> TypeOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypeAttr TypeOp::getConstantTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getConstantTypeAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::std::optional<::mlir::Type> TypeOp::getConstantType() {
  auto attr = getConstantTypeAttr();
  return attr ? ::std::optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::std::nullopt);
}

void TypeOp::setConstantTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getConstantTypeAttrName(), attr);
}

void TypeOp::setConstantType(::std::optional<::mlir::Type> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getConstantTypeAttrName(), ::mlir::TypeAttr::get(*attrValue));
    (*this)->removeAttr(getConstantTypeAttrName());
}

::mlir::Attribute TypeOp::removeConstantTypeAttr() {
  return (*this)->removeAttr(getConstantTypeAttrName());
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::TypeAttr constantType) {
  if (constantType) {
    odsState.addAttribute(getConstantTypeAttrName(odsState.name), constantType);
  }
  odsState.addTypes(result);
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::TypeAttr constantType) {
  if (constantType) {
    odsState.addAttribute(getConstantTypeAttrName(odsState.name), constantType);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TypeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_constantType;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getConstantTypeAttrName()) {
      tblgen_constantType = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps5(*this, tblgen_constantType, "constantType")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TypeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr constantTypeAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseCustomAttributeWithFallback(constantTypeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constantType",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("constantType");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if ((*this)->getAttr("constantType")) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getConstantTypeAttr());
  }
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::TypeOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::TypesOp definitions
//===----------------------------------------------------------------------===//

TypesOpAdaptor::TypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.types", odsAttrs.getContext());
}

TypesOpAdaptor::TypesOpAdaptor(TypesOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

std::pair<unsigned, unsigned> TypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange TypesOpAdaptor::getOperands() {
  return odsOperands;
}

::mlir::DictionaryAttr TypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TypesOpAdaptor::getConstantTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TypesOp::getConstantTypesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > TypesOpAdaptor::getConstantTypes() {
  auto attr = getConstantTypesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::LogicalResult TypesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_constantTypes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TypesOp::getConstantTypesAttrName(*odsOpName)) {
      tblgen_constantTypes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_constantTypes && !(((tblgen_constantTypes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_constantTypes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'pdl.types' op ""attribute 'constantTypes' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> TypesOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TypesOp::getConstantTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getConstantTypesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > TypesOp::getConstantTypes() {
  auto attr = getConstantTypesAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void TypesOp::setConstantTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getConstantTypesAttrName(), attr);
}

::mlir::Attribute TypesOp::removeConstantTypesAttr() {
  return (*this)->removeAttr(getConstantTypesAttrName());
}

void TypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::ArrayAttr constantTypes) {
  if (constantTypes) {
    odsState.addAttribute(getConstantTypesAttrName(odsState.name), constantTypes);
  }
  odsState.addTypes(result);
}

void TypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr constantTypes) {
  if (constantTypes) {
    odsState.addAttribute(getConstantTypesAttrName(odsState.name), constantTypes);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TypesOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_constantTypes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getConstantTypesAttrName()) {
      tblgen_constantTypes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps6(*this, tblgen_constantTypes, "constantTypes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TypesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr constantTypesAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseCustomAttributeWithFallback(constantTypesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constantTypes",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("constantTypes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if ((*this)->getAttr("constantTypes")) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getConstantTypesAttr());
  }
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::TypesOp)


#endif  // GET_OP_CLASSES

