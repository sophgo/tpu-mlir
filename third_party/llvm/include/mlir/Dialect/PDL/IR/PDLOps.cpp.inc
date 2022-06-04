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

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::getOperands() {
  return odsOperands;
}

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

::mlir::ValueRange ApplyNativeConstraintOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyNativeConstraintOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyNativeConstraintOpAdaptor::nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyNativeConstraintOp::nameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef ApplyNativeConstraintOpAdaptor::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyNativeConstraintOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.apply_native_constraint' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == ApplyNativeConstraintOp::nameAttrName(*odsOpName)) {
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

::mlir::Operation::operand_range ApplyNativeConstraintOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyNativeConstraintOp::argsMutable() {
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

::mlir::StringAttr ApplyNativeConstraintOp::nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, nameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyNativeConstraintOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

void ApplyNativeConstraintOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(nameAttrName(), attr);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(nameAttrName(odsState.name), name);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(nameAttrName(odsState.name), name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(nameAttrName(odsState.name), odsBuilder.getStringAttr(name));
}

void ApplyNativeConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(nameAttrName(odsState.name), odsBuilder.getStringAttr(name));
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
    if (namedAttrIt->getName() == nameAttrName()) {
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
  _odsPrinter.printAttributeWithoutType(nameAttr());
  _odsPrinter << "(";
  _odsPrinter << args();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << args().getTypes();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
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

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::getOperands() {
  return odsOperands;
}

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

::mlir::ValueRange ApplyNativeRewriteOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyNativeRewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyNativeRewriteOpAdaptor::nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyNativeRewriteOp::nameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef ApplyNativeRewriteOpAdaptor::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyNativeRewriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.apply_native_rewrite' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == ApplyNativeRewriteOp::nameAttrName(*odsOpName)) {
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

::mlir::Operation::operand_range ApplyNativeRewriteOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyNativeRewriteOp::argsMutable() {
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

::mlir::Operation::result_range ApplyNativeRewriteOp::results() {
  return getODSResults(0);
}

::mlir::StringAttr ApplyNativeRewriteOp::nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, nameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyNativeRewriteOp::name() {
  auto attr = nameAttr();
  return attr.getValue();
}

void ApplyNativeRewriteOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(nameAttrName(), attr);
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(nameAttrName(odsState.name), name);
  odsState.addTypes(results);
}

void ApplyNativeRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(nameAttrName(odsState.name), odsBuilder.getStringAttr(name));
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
    if (namedAttrIt->getName() == nameAttrName()) {
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
  if (succeeded(parser.parseOptionalLParen())) {

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
  if (succeeded(parser.parseOptionalColon())) {

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
  _odsPrinter.printAttributeWithoutType(nameAttr());
  if (!args().empty()) {
  _odsPrinter << "(";
  _odsPrinter << args();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << args().getTypes();
  _odsPrinter << ")";
  }
  if (!results().empty()) {
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << results().getTypes();
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
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

::mlir::ValueRange AttributeOpAdaptor::getOperands() {
  return odsOperands;
}

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

::mlir::Value AttributeOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr AttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute AttributeOpAdaptor::valueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AttributeOp::valueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::Attribute>();
    return attr;
}

::llvm::Optional<::mlir::Attribute> AttributeOpAdaptor::value() {
  auto attr = valueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

::mlir::LogicalResult AttributeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AttributeOp::valueAttrName(*odsOpName)) {
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

::mlir::Value AttributeOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange AttributeOp::typeMutable() {
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

::mlir::Value AttributeOp::attr() {
  return *getODSResults(0).begin();
}

::mlir::Attribute AttributeOp::valueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, valueAttrName()).dyn_cast_or_null<::mlir::Attribute>();
}

::llvm::Optional<::mlir::Attribute> AttributeOp::value() {
  auto attr = valueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

void AttributeOp::valueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(valueAttrName(), attr);
}

::mlir::Attribute AttributeOp::removeValueAttr() {
  return (*this)->removeAttr(valueAttrName());
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value type) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), type,
            Attribute());
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute attr) {
      build(odsBuilder, odsState, odsBuilder.getType<AttributeType>(), Value(), attr);
    
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attr, /*optional*/::mlir::Value type, /*optional*/::mlir::Attribute value) {
  if (type)
    odsState.addOperands(type);
  if (value) {
  odsState.addAttribute(valueAttrName(odsState.name), value);
  }
  odsState.addTypes(attr);
}

void AttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type, /*optional*/::mlir::Attribute value) {
  if (type)
    odsState.addOperands(type);
  if (value) {
  odsState.addAttribute(valueAttrName(odsState.name), value);
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
    else if (namedAttrIt->getName() == valueAttrName()) {
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  ::mlir::Attribute valueAttr;
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }
  if (succeeded(parser.parseOptionalEqual())) {

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}, "value", result.attributes))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (type()) {
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  if (::mlir::Value value = type())
    _odsPrinter << value;
  }
  if ((*this)->getAttr("value")) {
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(valueAttr());
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
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

::mlir::ValueRange EraseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> EraseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EraseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EraseOpAdaptor::operation() {
  return *getODSOperands(0).begin();
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

::mlir::Value EraseOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EraseOp::operationMutable() {
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

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation) {
  odsState.addOperands(operation);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation) {
  odsState.addOperands(operation);
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
  ::mlir::OpAsmParser::UnresolvedOperand operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << operation();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
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

::mlir::ValueRange OperandOpAdaptor::getOperands() {
  return odsOperands;
}

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

::mlir::Value OperandOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
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

::mlir::Value OperandOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange OperandOp::typeMutable() {
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

::mlir::Value OperandOp::val() {
  return *getODSResults(0).begin();
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, odsBuilder.getType<ValueType>(), Value());
    
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  odsState.addTypes(val);
}

void OperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (type()) {
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  if (::mlir::Value value = type())
    _odsPrinter << value;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
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

::mlir::ValueRange OperandsOpAdaptor::getOperands() {
  return odsOperands;
}

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

::mlir::Value OperandsOpAdaptor::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
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

::mlir::Value OperandsOp::type() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange OperandsOp::typeMutable() {
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

::mlir::Value OperandsOp::val() {
  return *getODSResults(0).begin();
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState, RangeType::get(odsBuilder.getType<ValueType>()),
            Value());
    
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
  odsState.addTypes(val);
}

void OperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value type) {
  if (type)
    odsState.addOperands(type);
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> typeOperands;
  ::llvm::SMLoc typeOperandsLoc;
  (void)typeOperandsLoc;
  if (succeeded(parser.parseOptionalColon())) {

  {
    typeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      typeOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::ValueType>());
  ::mlir::Type odsBuildableType1 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(typeOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperandsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (type()) {
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  if (::mlir::Value value = type())
    _odsPrinter << value;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
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

::mlir::ValueRange OperationOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OperationOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, OperationOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange OperationOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange OperationOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::ValueRange OperationOpAdaptor::attributes() {
  return getODSOperands(1);
}

::mlir::ValueRange OperationOpAdaptor::types() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr OperationOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr OperationOpAdaptor::nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, OperationOp::nameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > OperationOpAdaptor::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr OperationOpAdaptor::attributeNamesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, OperationOp::attributeNamesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr OperationOpAdaptor::attributeNames() {
  auto attr = attributeNamesAttr();
  return attr;
}

::mlir::LogicalResult OperationOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_attributeNames;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.operation' op ""requires attribute 'attributeNames'");
    if (namedAttrIt->getName() == OperationOp::attributeNamesAttrName(*odsOpName)) {
      tblgen_attributeNames = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.operation' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == OperationOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == OperationOp::nameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'pdl.operation' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.operation' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_attributeNames && !(((tblgen_attributeNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_attributeNames.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'pdl.operation' op ""attribute 'attributeNames' failed to satisfy constraint: string array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range OperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OperationOp::operands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range OperationOp::attributes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range OperationOp::types() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange OperationOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange OperationOp::attributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange OperationOp::typesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
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

::mlir::Value OperationOp::op() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr OperationOp::nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, nameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > OperationOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr OperationOp::attributeNamesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, attributeNamesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr OperationOp::attributeNames() {
  auto attr = attributeNamesAttr();
  return attr;
}

void OperationOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(nameAttrName(), attr);
}

void OperationOp::attributeNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(attributeNamesAttrName(), attr);
}

::mlir::Attribute OperationOp::removeNameAttr() {
  return (*this)->removeAttr(nameAttrName());
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Optional<StringRef> name, ValueRange operandValues, ArrayRef<StringRef> attrNames, ValueRange attrValues, ValueRange resultTypes) {
      auto nameAttr = name ? odsBuilder.getStringAttr(*name) : StringAttr();
      build(odsBuilder, odsState, odsBuilder.getType<OperationType>(), nameAttr,
            operandValues, attrValues, odsBuilder.getStrArrayAttr(attrNames),
            resultTypes);
    
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type op, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  odsState.addAttribute(attributeNamesAttrName(odsState.name), attributeNames);
  odsState.addTypes(op);
}

void OperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange operands, ::mlir::ValueRange attributes, ::mlir::ArrayAttr attributeNames, ::mlir::ValueRange types) {
  odsState.addOperands(operands);
  odsState.addOperands(attributes);
  odsState.addOperands(types);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(operands.size()), static_cast<int32_t>(attributes.size()), static_cast<int32_t>(types.size())}));
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  odsState.addAttribute(attributeNamesAttrName(odsState.name), attributeNames);
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
  ::mlir::Attribute tblgen_attributeNames;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'attributeNames'");
    if (namedAttrIt->getName() == attributeNamesAttrName()) {
      tblgen_attributeNames = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == nameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps2(*this, tblgen_attributeNames, "attributeNames")))
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
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attributesOperands;
  ::llvm::SMLoc attributesOperandsLoc;
  (void)attributesOperandsLoc;
  ::mlir::ArrayAttr attributeNamesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> typesOperands;
  ::llvm::SMLoc typesOperandsLoc;
  (void)typesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> typesTypes;

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (nameAttr) {
  }
  if (succeeded(parser.parseOptionalLParen())) {

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    attributesOperandsLoc = parser.getCurrentLocation();
    if (parseOperationOpAttributes(parser, attributesOperands, attributeNamesAttr))
      return ::mlir::failure();
    result.addAttribute("attributeNames", attributeNamesAttr);
  }
  if (succeeded(parser.parseOptionalArrow())) {
  if (parser.parseLParen())
    return ::mlir::failure();

  typesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(typesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(typesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(operandsOperands.size()), static_cast<int32_t>(attributesOperands.size()), static_cast<int32_t>(typesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attributesOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(typesOperands, typesTypes, typesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OperationOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("name")) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(nameAttr());
  }
  if (!operands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << operands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << operands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  printOperationOpAttributes(_odsPrinter, *this, attributes(), attributeNamesAttr());
  if (!types().empty()) {
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ' << "(";
  _odsPrinter << types();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << types().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "name", "attributeNames"});
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

::mlir::ValueRange PatternOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PatternOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PatternOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr PatternOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr PatternOpAdaptor::benefitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PatternOp::benefitAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint16_t PatternOpAdaptor::benefit() {
  auto attr = benefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr PatternOpAdaptor::sym_nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, PatternOp::sym_nameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > PatternOpAdaptor::sym_name() {
  auto attr = sym_nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange PatternOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &PatternOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult PatternOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_benefit;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.pattern' op ""requires attribute 'benefit'");
    if (namedAttrIt->getName() == PatternOp::benefitAttrName(*odsOpName)) {
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
    else if (namedAttrIt->getName() == PatternOp::sym_nameAttrName(*odsOpName)) {
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

::mlir::Region &PatternOp::body() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr PatternOp::benefitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, benefitAttrName()).cast<::mlir::IntegerAttr>();
}

uint16_t PatternOp::benefit() {
  auto attr = benefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::StringAttr PatternOp::sym_nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, sym_nameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > PatternOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void PatternOp::benefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(benefitAttrName(), attr);
}

void PatternOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(sym_nameAttrName(), attr);
}

::mlir::Attribute PatternOp::removeSym_nameAttr() {
  return (*this)->removeAttr(sym_nameAttrName());
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(benefitAttrName(odsState.name), benefit);
  if (sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(benefitAttrName(odsState.name), benefit);
  if (sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(benefitAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  }
  (void)odsState.addRegion();
}

void PatternOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint16_t benefit, /*optional*/::mlir::StringAttr sym_name) {
  odsState.addAttribute(benefitAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  if (sym_name) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
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
    if (namedAttrIt->getName() == benefitAttrName()) {
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
    else if (namedAttrIt->getName() == sym_nameAttrName()) {
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
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLOps0(*this, region, "body", index++)))
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
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

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

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();

  if (bodyRegion->empty()) bodyRegion->emplaceBlock();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void PatternOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("sym_name")) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(sym_nameAttr().getValue());
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "benefit";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(benefitAttr());
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name", "benefit"});
  _odsPrinter << ' ';
  _odsPrinter.printRegion(body());
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::PatternOp)

namespace mlir {
namespace pdl {

//===----------------------------------------------------------------------===//
// ::mlir::pdl::ReplaceOp definitions
//===----------------------------------------------------------------------===//

ReplaceOpAdaptor::ReplaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl.replace", odsAttrs.getContext());
}

ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ReplaceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReplaceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReplaceOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange ReplaceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOpAdaptor::operation() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReplaceOpAdaptor::replOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ReplaceOpAdaptor::replValues() {
  return getODSOperands(2);
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
    if (namedAttrIt->getName() == ReplaceOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'pdl.replace' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range ReplaceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOp::operation() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReplaceOp::replOperation() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ReplaceOp::replValues() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ReplaceOp::operationMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReplaceOp::replOperationMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReplaceOp::replValuesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
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

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operation, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operation, /*optional*/::mlir::Value replOperation, ::mlir::ValueRange replValues) {
  odsState.addOperands(operation);
  if (replOperation)
    odsState.addOperands(replOperation);
  odsState.addOperands(replValues);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, (replOperation ? 1 : 0), static_cast<int32_t>(replValues.size())}));
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
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
  ::mlir::OpAsmParser::UnresolvedOperand operationRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operationOperands(operationRawOperands);  ::llvm::SMLoc operationOperandsLoc;
  (void)operationOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> replValuesTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replOperationOperands;
  ::llvm::SMLoc replOperationOperandsLoc;
  (void)replOperationOperandsLoc;

  operationOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operationRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

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
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      replOperationOperands.push_back(operand);
    }
  }
  if (!replOperationOperands.empty()) {
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(replOperationOperands.size()), static_cast<int32_t>(replValuesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(operationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replOperationOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << operation();
  _odsPrinter << ' ' << "with";
  if (!replValues().empty()) {
  _odsPrinter << "(";
  _odsPrinter << replValues();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << replValues().getTypes();
  _odsPrinter << ")";
  }
  if (replOperation()) {
  _odsPrinter << ' ';
  if (::mlir::Value value = replOperation())
    _odsPrinter << value;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
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

::mlir::ValueRange ResultOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultOpAdaptor::parent() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResultOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ResultOpAdaptor::indexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ResultOp::indexAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t ResultOpAdaptor::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ResultOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.result' op ""requires attribute 'index'");
    if (namedAttrIt->getName() == ResultOp::indexAttrName(*odsOpName)) {
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

::mlir::Value ResultOp::parent() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResultOp::parentMutable() {
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

::mlir::Value ResultOp::val() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ResultOp::indexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, indexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t ResultOp::index() {
  auto attr = indexAttr();
  return attr.getValue().getZExtValue();
}

void ResultOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(indexAttrName(), attr);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, ::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  odsState.addAttribute(indexAttrName(odsState.name), index);
  odsState.addTypes(val);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, ::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  odsState.addAttribute(indexAttrName(odsState.name), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, uint32_t index) {
  odsState.addOperands(parent);
  odsState.addAttribute(indexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(val);
}

void ResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, uint32_t index) {
  odsState.addOperands(parent);
  odsState.addAttribute(indexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
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
    if (namedAttrIt->getName() == indexAttrName()) {
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
  if (parser.resolveOperands(parentOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(indexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << parent();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
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

::mlir::ValueRange ResultsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResultsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResultsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResultsOpAdaptor::parent() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResultsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ResultsOpAdaptor::indexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ResultsOp::indexAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint32_t> ResultsOpAdaptor::index() {
  auto attr = indexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult ResultsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ResultsOp::indexAttrName(*odsOpName)) {
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

::mlir::Value ResultsOp::parent() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResultsOp::parentMutable() {
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

::mlir::Value ResultsOp::val() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ResultsOp::indexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, indexAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> ResultsOp::index() {
  auto attr = indexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void ResultsOp::indexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(indexAttrName(), attr);
}

::mlir::Attribute ResultsOp::removeIndexAttr() {
  return (*this)->removeAttr(indexAttrName());
}

void ResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type val, ::mlir::Value parent, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  if (index) {
  odsState.addAttribute(indexAttrName(odsState.name), index);
  }
  odsState.addTypes(val);
}

void ResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value parent, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(parent);
  if (index) {
  odsState.addAttribute(indexAttrName(odsState.name), index);
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
    else if (namedAttrIt->getName() == indexAttrName()) {
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
    if (parseResult.hasValue() && failed(*parseResult))
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
  if (parser.resolveOperands(parentOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResultsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("index")) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(indexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << parent();
  _odsPrinter << ' ';
  printResultsValueType(_odsPrinter, *this, indexAttr(), val().getType());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
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

::mlir::ValueRange RewriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, RewriteOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange RewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOpAdaptor::root() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange RewriteOpAdaptor::externalArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr RewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr RewriteOpAdaptor::nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RewriteOp::nameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > RewriteOpAdaptor::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange RewriteOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &RewriteOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult RewriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl.rewrite' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RewriteOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RewriteOp::nameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'pdl.rewrite' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl.rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RewriteOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range RewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RewriteOp::root() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range RewriteOp::externalArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RewriteOp::rootMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RewriteOp::externalArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
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

::mlir::Region &RewriteOp::body() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr RewriteOp::nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, nameAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > RewriteOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void RewriteOp::nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(nameAttrName(), attr);
}

::mlir::Attribute RewriteOp::removeNameAttr() {
  return (*this)->removeAttr(nameAttrName());
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs) {
  if (root)
    odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(root ? 1 : 0), static_cast<int32_t>(externalArgs.size())}));
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
}

void RewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value root, /*optional*/::mlir::StringAttr name, ::mlir::ValueRange externalArgs) {
  if (root)
    odsState.addOperands(root);
  odsState.addOperands(externalArgs);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(root ? 1 : 0), static_cast<int32_t>(externalArgs.size())}));
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
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
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == nameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
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
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLOps1(*this, region, "body", index++)))
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
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  {
    rootOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      rootOperands.push_back(operand);
    }
  }
  if (!rootOperands.empty()) {
  }
  if (succeeded(parser.parseOptionalKeyword("with"))) {

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalLParen())) {

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
     auto parseResult = parser.parseOptionalRegion(*bodyRegion);
     if (parseResult.hasValue() && failed(*parseResult))
       return ::mlir::failure();
  }
  if (!bodyRegion->empty()) {

  if (bodyRegion->empty()) bodyRegion->emplaceBlock();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(rootOperands.size()), static_cast<int32_t>(externalArgsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(rootOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(externalArgsOperands, externalArgsTypes, externalArgsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RewriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (root()) {
  _odsPrinter << ' ';
  if (::mlir::Value value = root())
    _odsPrinter << value;
  }
  if ((*this)->getAttr("name")) {
  _odsPrinter << ' ' << "with";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(nameAttr());
  if (!externalArgs().empty()) {
  _odsPrinter << "(";
  _odsPrinter << externalArgs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << externalArgs().getTypes();
  _odsPrinter << ")";
  }
  }
  if (!body().empty()) {
  _odsPrinter << ' ';
  _odsPrinter.printRegion(body());
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "name"});
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

::mlir::ValueRange TypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr TypeOpAdaptor::typeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TypeOp::typeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
    return attr;
}

::llvm::Optional<::mlir::Type> TypeOpAdaptor::type() {
  auto attr = typeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::LogicalResult TypeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TypeOp::typeAttrName(*odsOpName)) {
      tblgen_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'pdl.type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
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

::mlir::Value TypeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::TypeAttr TypeOp::typeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, typeAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> TypeOp::type() {
  auto attr = typeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

void TypeOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(typeAttrName(), attr);
}

::mlir::Attribute TypeOp::removeTypeAttr() {
  return (*this)->removeAttr(typeAttrName());
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::TypeAttr type) {
  if (type) {
  odsState.addAttribute(typeAttrName(odsState.name), type);
  }
  odsState.addTypes(result);
}

void TypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::TypeAttr type) {
  if (type) {
  odsState.addAttribute(typeAttrName(odsState.name), type);
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
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == typeAttrName()) {
      tblgen_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps5(*this, tblgen_type, "type")))
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
  ::mlir::TypeAttr typeAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"type"});
  if ((*this)->getAttr("type")) {
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(typeAttr());
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

::mlir::ValueRange TypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TypesOpAdaptor::typesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TypesOp::typesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TypesOpAdaptor::types() {
  auto attr = typesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult TypesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TypesOp::typesAttrName(*odsOpName)) {
      tblgen_types = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_types && !(((tblgen_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'pdl.types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
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

::mlir::Value TypesOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TypesOp::typesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, typesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TypesOp::types() {
  auto attr = typesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void TypesOp::typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(typesAttrName(), attr);
}

::mlir::Attribute TypesOp::removeTypesAttr() {
  return (*this)->removeAttr(typesAttrName());
}

void TypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::ArrayAttr types) {
  if (types) {
  odsState.addAttribute(typesAttrName(odsState.name), types);
  }
  odsState.addTypes(result);
}

void TypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::ArrayAttr types) {
  if (types) {
  odsState.addAttribute(typesAttrName(odsState.name), types);
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
  ::mlir::Attribute tblgen_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == typesAttrName()) {
      tblgen_types = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLOps6(*this, tblgen_types, "types")))
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
  ::mlir::ArrayAttr typesAttr;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalColon())) {

  if (parser.parseCustomAttributeWithFallback(typesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "types",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"types"});
  if ((*this)->getAttr("types")) {
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(typesAttr());
  }
}

} // namespace pdl
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl::TypesOp)


#endif  // GET_OP_CLASSES

