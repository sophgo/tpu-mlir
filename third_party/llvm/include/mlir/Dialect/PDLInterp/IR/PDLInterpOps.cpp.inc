/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl_interp::ApplyConstraintOp,
::mlir::pdl_interp::ApplyRewriteOp,
::mlir::pdl_interp::AreEqualOp,
::mlir::pdl_interp::BranchOp,
::mlir::pdl_interp::CheckAttributeOp,
::mlir::pdl_interp::CheckOperandCountOp,
::mlir::pdl_interp::CheckOperationNameOp,
::mlir::pdl_interp::CheckResultCountOp,
::mlir::pdl_interp::CheckTypeOp,
::mlir::pdl_interp::CheckTypesOp,
::mlir::pdl_interp::ContinueOp,
::mlir::pdl_interp::CreateAttributeOp,
::mlir::pdl_interp::CreateOperationOp,
::mlir::pdl_interp::CreateTypeOp,
::mlir::pdl_interp::CreateTypesOp,
::mlir::pdl_interp::EraseOp,
::mlir::pdl_interp::ExtractOp,
::mlir::pdl_interp::FinalizeOp,
::mlir::pdl_interp::ForEachOp,
::mlir::pdl_interp::FuncOp,
::mlir::pdl_interp::GetAttributeOp,
::mlir::pdl_interp::GetAttributeTypeOp,
::mlir::pdl_interp::GetDefiningOpOp,
::mlir::pdl_interp::GetOperandOp,
::mlir::pdl_interp::GetOperandsOp,
::mlir::pdl_interp::GetResultOp,
::mlir::pdl_interp::GetResultsOp,
::mlir::pdl_interp::GetUsersOp,
::mlir::pdl_interp::GetValueTypeOp,
::mlir::pdl_interp::InferredTypesOp,
::mlir::pdl_interp::IsNotNullOp,
::mlir::pdl_interp::RecordMatchOp,
::mlir::pdl_interp::ReplaceOp,
::mlir::pdl_interp::SwitchAttributeOp,
::mlir::pdl_interp::SwitchOperandCountOp,
::mlir::pdl_interp::SwitchOperationNameOp,
::mlir::pdl_interp::SwitchResultCountOp,
::mlir::pdl_interp::SwitchTypeOp,
::mlir::pdl_interp::SwitchTypesOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl_interp {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::PDLType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::AttributeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::TypeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::ValueType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::PDLType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of pdl type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::ValueType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::OperationType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Operation *` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!attr.cast<::mlir::IntegerAttr>().getValue().isNegative())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::SymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!attr.cast<::mlir::IntegerAttr>().getValue().isNegative())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type-array array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_PDLInterpOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItemsOrMore(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with at least 1 blocks";
  }
  return ::mlir::success();
}
} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyConstraintOp definitions
//===----------------------------------------------------------------------===//

ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(ApplyConstraintOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ApplyConstraintOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyConstraintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyConstraintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyConstraintOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyConstraintOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyConstraintOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyConstraintOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyConstraintOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc, "'pdl_interp.apply_constraint' op ""requires attribute 'name'");

    if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyConstraintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyConstraintOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyConstraintOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyConstraintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *ApplyConstraintOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *ApplyConstraintOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::StringAttr ApplyConstraintOp::getNameAttr() {
  return (*this)->getAttr(getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyConstraintOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyConstraintOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyConstraintOp::verifyInvariantsImpl() {
  {
    auto tblgen_name = (*this)->getAttr(getNameAttrName());
    if (!tblgen_name)
      return emitOpError("requires attribute 'name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyConstraintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ApplyConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

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
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyConstraintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "(";
  _odsPrinter << getArgs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyConstraintOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyRewriteOp definitions
//===----------------------------------------------------------------------===//

ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(ApplyRewriteOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ApplyRewriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyRewriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ApplyRewriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ApplyRewriteOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ApplyRewriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyRewriteOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyRewriteOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyRewriteOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc, "'pdl_interp.apply_rewrite' op ""requires attribute 'name'");

    if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.apply_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyRewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyRewriteOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyRewriteOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ApplyRewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ApplyRewriteOp::getResults() {
  return getODSResults(0);
}

::mlir::StringAttr ApplyRewriteOp::getNameAttr() {
  return (*this)->getAttr(getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyRewriteOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyRewriteOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyRewriteOp::verifyInvariantsImpl() {
  {
    auto tblgen_name = (*this)->getAttr(getNameAttrName());
    if (!tblgen_name)
      return emitOpError("requires attribute 'name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyRewriteOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ApplyRewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> argsTypes;
  ::mlir::SmallVector<::mlir::Type, 1> resultsTypes;

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

void ApplyRewriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyRewriteOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::AreEqualOp definitions
//===----------------------------------------------------------------------===//

AreEqualOpAdaptor::AreEqualOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AreEqualOpAdaptor::AreEqualOpAdaptor(AreEqualOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange AreEqualOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AreEqualOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AreEqualOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AreEqualOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AreEqualOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AreEqualOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AreEqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AreEqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AreEqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AreEqualOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AreEqualOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AreEqualOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AreEqualOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AreEqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AreEqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *AreEqualOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *AreEqualOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AreEqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AreEqualOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult AreEqualOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AreEqualOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const ::mlir::Type>(::llvm::ArrayRef<::mlir::Type>(lhsTypes), ::llvm::ArrayRef<::mlir::Type>(lhsTypes)), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AreEqualOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void AreEqualOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::AreEqualOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::BranchOp definitions
//===----------------------------------------------------------------------===//

BranchOpAdaptor::BranchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

BranchOpAdaptor::BranchOpAdaptor(BranchOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange BranchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BranchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BranchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BranchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BranchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BranchOp::getDest() {
  return (*this)->getSuccessor(0);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BranchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult BranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void BranchOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::BranchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckAttributeOp definitions
//===----------------------------------------------------------------------===//

CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(CheckAttributeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CheckAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckAttributeOpAdaptor::getAttribute() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CheckAttributeOpAdaptor::getConstantValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("constantValue").cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute CheckAttributeOpAdaptor::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

::mlir::LogicalResult CheckAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_constantValue = odsAttrs.get("constantValue");
    if (!tblgen_constantValue)
      return emitError(loc, "'pdl_interp.check_attribute' op ""requires attribute 'constantValue'");

    if (tblgen_constantValue && !((true)))
      return emitError(loc, "'pdl_interp.check_attribute' op ""attribute 'constantValue' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckAttributeOp::getAttribute() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckAttributeOp::getAttributeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckAttributeOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckAttributeOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::Attribute CheckAttributeOp::getConstantValueAttr() {
  return (*this)->getAttr(getConstantValueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute CheckAttributeOp::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

void CheckAttributeOp::setConstantValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getConstantValueAttrName(), attr);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.addAttribute(getConstantValueAttrName(odsState.name), constantValue);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.addAttribute(getConstantValueAttrName(odsState.name), constantValue);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckAttributeOp::verifyInvariantsImpl() {
  {
    auto tblgen_constantValue = (*this)->getAttr(getConstantValueAttrName());
    if (!tblgen_constantValue)
      return emitOpError("requires attribute 'constantValue'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_constantValue, "constantValue")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand attributeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> attributeOperands(attributeRawOperands);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::Attribute constantValueAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseAttribute(constantValueAttr, ::mlir::Type{}, "constantValue", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getConstantValueAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"constantValue"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperandCountOp definitions
//===----------------------------------------------------------------------===//

CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(CheckOperandCountOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CheckOperandCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckOperandCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckOperandCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperandCountOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckOperandCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckOperandCountOpAdaptor::getCountAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("count").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t CheckOperandCountOpAdaptor::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOpAdaptor::getCompareAtLeastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("compareAtLeast").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool CheckOperandCountOpAdaptor::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
  return attr != nullptr;
}

::mlir::LogicalResult CheckOperandCountOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_count = odsAttrs.get("count");
    if (!tblgen_count)
      return emitError(loc, "'pdl_interp.check_operand_count' op ""requires attribute 'count'");

    if (tblgen_count && !((((tblgen_count.isa<::mlir::IntegerAttr>())) && ((tblgen_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_count.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  {
    auto tblgen_compareAtLeast = odsAttrs.get("compareAtLeast");
    if (tblgen_compareAtLeast && !((tblgen_compareAtLeast.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckOperandCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckOperandCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperandCountOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckOperandCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckOperandCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckOperandCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckOperandCountOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckOperandCountOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::IntegerAttr CheckOperandCountOp::getCountAttr() {
  return (*this)->getAttr(getCountAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t CheckOperandCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOp::getCompareAtLeastAttr() {
  return (*this)->getAttr(getCompareAtLeastAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckOperandCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
  return attr != nullptr;
}

void CheckOperandCountOp::setCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCountAttrName(), attr);
}

void CheckOperandCountOp::setCompareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCompareAtLeastAttrName(), attr);
}

::mlir::Attribute CheckOperandCountOp::removeCompareAtLeastAttr() {
  return (*this)->removeAttr(getCompareAtLeastAttrName());
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), count);
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), count);
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckOperandCountOp::verifyInvariantsImpl() {
  {
    auto tblgen_count = (*this)->getAttr(getCountAttrName());
    if (!tblgen_count)
      return emitOpError("requires attribute 'count'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_count, "count")))
      return ::mlir::failure();
  }
  {
    auto tblgen_compareAtLeast = (*this)->getAttr(getCompareAtLeastAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_compareAtLeast, "compareAtLeast")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckOperandCountOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.addAttribute("compareAtLeast", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32), "count",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperandCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  if ((*this)->getAttr("compareAtLeast")) {
  _odsPrinter << ' ' << "at_least";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"compareAtLeast", "count"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperandCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperationNameOp definitions
//===----------------------------------------------------------------------===//

CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(CheckOperationNameOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CheckOperationNameOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckOperationNameOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckOperationNameOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperationNameOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckOperationNameOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CheckOperationNameOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CheckOperationNameOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult CheckOperationNameOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc, "'pdl_interp.check_operation_name' op ""requires attribute 'name'");

    if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.check_operation_name' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckOperationNameOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckOperationNameOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckOperationNameOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckOperationNameOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckOperationNameOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckOperationNameOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckOperationNameOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckOperationNameOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::StringAttr CheckOperationNameOp::getNameAttr() {
  return (*this)->getAttr(getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CheckOperationNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CheckOperationNameOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckOperationNameOp::verifyInvariantsImpl() {
  {
    auto tblgen_name = (*this)->getAttr(getNameAttrName());
    if (!tblgen_name)
      return emitOpError("requires attribute 'name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckOperationNameOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperationNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperationNameOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckResultCountOp definitions
//===----------------------------------------------------------------------===//

CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(CheckResultCountOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CheckResultCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckResultCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckResultCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckResultCountOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckResultCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckResultCountOpAdaptor::getCountAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("count").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t CheckResultCountOpAdaptor::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOpAdaptor::getCompareAtLeastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("compareAtLeast").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool CheckResultCountOpAdaptor::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
  return attr != nullptr;
}

::mlir::LogicalResult CheckResultCountOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_count = odsAttrs.get("count");
    if (!tblgen_count)
      return emitError(loc, "'pdl_interp.check_result_count' op ""requires attribute 'count'");

    if (tblgen_count && !((((tblgen_count.isa<::mlir::IntegerAttr>())) && ((tblgen_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_count.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  {
    auto tblgen_compareAtLeast = odsAttrs.get("compareAtLeast");
    if (tblgen_compareAtLeast && !((tblgen_compareAtLeast.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckResultCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckResultCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckResultCountOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckResultCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckResultCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckResultCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckResultCountOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckResultCountOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::IntegerAttr CheckResultCountOp::getCountAttr() {
  return (*this)->getAttr(getCountAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t CheckResultCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOp::getCompareAtLeastAttr() {
  return (*this)->getAttr(getCompareAtLeastAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckResultCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
  return attr != nullptr;
}

void CheckResultCountOp::setCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCountAttrName(), attr);
}

void CheckResultCountOp::setCompareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCompareAtLeastAttrName(), attr);
}

::mlir::Attribute CheckResultCountOp::removeCompareAtLeastAttr() {
  return (*this)->removeAttr(getCompareAtLeastAttrName());
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), count);
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), count);
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), compareAtLeast);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
  odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckResultCountOp::verifyInvariantsImpl() {
  {
    auto tblgen_count = (*this)->getAttr(getCountAttrName());
    if (!tblgen_count)
      return emitOpError("requires attribute 'count'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_count, "count")))
      return ::mlir::failure();
  }
  {
    auto tblgen_compareAtLeast = (*this)->getAttr(getCompareAtLeastAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_compareAtLeast, "compareAtLeast")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckResultCountOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.addAttribute("compareAtLeast", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32), "count",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckResultCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  if ((*this)->getAttr("compareAtLeast")) {
  _odsPrinter << ' ' << "at_least";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"compareAtLeast", "count"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckResultCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypeOp definitions
//===----------------------------------------------------------------------===//

CheckTypeOpAdaptor::CheckTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CheckTypeOpAdaptor::CheckTypeOpAdaptor(CheckTypeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CheckTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypeOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CheckTypeOpAdaptor::getTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type CheckTypeOpAdaptor::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::LogicalResult CheckTypeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_type = odsAttrs.get("type");
    if (!tblgen_type)
      return emitError(loc, "'pdl_interp.check_type' op ""requires attribute 'type'");

    if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
      return emitError(loc, "'pdl_interp.check_type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypeOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckTypeOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckTypeOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::TypeAttr CheckTypeOp::getTypeAttr() {
  return (*this)->getAttr(getTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type CheckTypeOp::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void CheckTypeOp::setTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTypeAttrName(), attr);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckTypeOp::verifyInvariantsImpl() {
  {
    auto tblgen_type = (*this)->getAttr(getTypeAttrName());
    if (!tblgen_type)
      return emitOpError("requires attribute 'type'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_type, "type")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::TypeAttr typeAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypeAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"type"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypesOp definitions
//===----------------------------------------------------------------------===//

CheckTypesOpAdaptor::CheckTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CheckTypesOpAdaptor::CheckTypesOpAdaptor(CheckTypesOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CheckTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CheckTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CheckTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypesOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CheckTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CheckTypesOpAdaptor::getTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CheckTypesOpAdaptor::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

::mlir::LogicalResult CheckTypesOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_types = odsAttrs.get("types");
    if (!tblgen_types)
      return emitError(loc, "'pdl_interp.check_types' op ""requires attribute 'types'");

    if (tblgen_types && !(((tblgen_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
      return emitError(loc, "'pdl_interp.check_types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CheckTypesOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CheckTypesOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckTypesOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckTypesOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::ArrayAttr CheckTypesOp::getTypesAttr() {
  return (*this)->getAttr(getTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CheckTypesOp::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

void CheckTypesOp::setTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTypesAttrName(), attr);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute(getTypesAttrName(odsState.name), types);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addAttribute(getTypesAttrName(odsState.name), types);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckTypesOp::verifyInvariantsImpl() {
  {
    auto tblgen_types = (*this)->getAttr(getTypesAttrName());
    if (!tblgen_types)
      return emitOpError("requires attribute 'types'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_types, "types")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr typesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("are"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "types",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "are";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypesAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"types"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypesOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ContinueOp definitions
//===----------------------------------------------------------------------===//

ContinueOpAdaptor::ContinueOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ContinueOpAdaptor::ContinueOpAdaptor(ContinueOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ContinueOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ContinueOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ContinueOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ContinueOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ContinueOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ContinueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ContinueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ContinueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ContinueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ContinueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ContinueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ContinueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ContinueOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult ContinueOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ContinueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void ContinueOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void ContinueOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ContinueOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateAttributeOp definitions
//===----------------------------------------------------------------------===//

CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(CreateAttributeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CreateAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CreateAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CreateAttributeOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute CreateAttributeOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::LogicalResult CreateAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_value = odsAttrs.get("value");
    if (!tblgen_value)
      return emitError(loc, "'pdl_interp.create_attribute' op ""requires attribute 'value'");

    if (tblgen_value && !((true)))
      return emitError(loc, "'pdl_interp.create_attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateAttributeOp::getAttribute() {
  return *getODSResults(0).begin();
}

::mlir::Attribute CreateAttributeOp::getValueAttr() {
  return (*this)->getAttr(getValueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute CreateAttributeOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void CreateAttributeOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::AttributeType>(), value);
    
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(attribute);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateAttributeOp::verifyInvariantsImpl() {
  {
    auto tblgen_value = (*this)->getAttr(getValueAttrName());
    if (!tblgen_value)
      return emitOpError("requires attribute 'value'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_value, "value")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}, "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getValueAttr());
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

void CreateAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateOperationOp definitions
//===----------------------------------------------------------------------===//

CreateOperationOpAdaptor::CreateOperationOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CreateOperationOpAdaptor::CreateOperationOpAdaptor(CreateOperationOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CreateOperationOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateOperationOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange CreateOperationOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CreateOperationOpAdaptor::getInputOperands() {
  return getODSOperands(0);
}

::mlir::ValueRange CreateOperationOpAdaptor::getInputAttributes() {
  return getODSOperands(1);
}

::mlir::ValueRange CreateOperationOpAdaptor::getInputResultTypes() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr CreateOperationOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CreateOperationOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CreateOperationOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOpAdaptor::getInputAttributeNamesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("inputAttributeNames").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CreateOperationOpAdaptor::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

::mlir::LogicalResult CreateOperationOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'pdl_interp.create_operation' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'pdl_interp.create_operation' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'name'");

    if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_inputAttributeNames = odsAttrs.get("inputAttributeNames");
    if (!tblgen_inputAttributeNames)
      return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'inputAttributeNames'");

    if (tblgen_inputAttributeNames && !(((tblgen_inputAttributeNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_inputAttributeNames.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
      return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inputAttributeNames' failed to satisfy constraint: string array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(getOperandSegmentSizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range CreateOperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateOperationOp::getInputOperands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range CreateOperationOp::getInputAttributes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CreateOperationOp::getInputResultTypes() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CreateOperationOp::getInputOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputAttributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputResultTypesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateOperationOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateOperationOp::getResultOp() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr CreateOperationOp::getNameAttr() {
  return (*this)->getAttr(getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CreateOperationOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNamesAttr() {
  return (*this)->getAttr(getInputAttributeNamesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

void CreateOperationOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CreateOperationOp::setInputAttributeNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInputAttributeNamesAttrName(), attr);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange types, ValueRange operands, ValueRange attributes, ArrayAttr attributeNames) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types);
    
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateOperationOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(getOperandSegmentSizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_name = (*this)->getAttr(getNameAttrName());
    if (!tblgen_name)
      return emitOpError("requires attribute 'name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
      return ::mlir::failure();
  }
  {
    auto tblgen_inputAttributeNames = (*this)->getAttr(getInputAttributeNamesAttrName());
    if (!tblgen_inputAttributeNames)
      return emitOpError("requires attribute 'inputAttributeNames'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_inputAttributeNames, "inputAttributeNames")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateOperationOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateOperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputOperandsOperands;
  ::llvm::SMLoc inputOperandsOperandsLoc;
  (void)inputOperandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputOperandsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputAttributesOperands;
  ::llvm::SMLoc inputAttributesOperandsLoc;
  (void)inputAttributesOperandsLoc;
  ::mlir::ArrayAttr inputAttributeNamesAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputResultTypesOperands;
  ::llvm::SMLoc inputResultTypesOperandsLoc;
  (void)inputResultTypesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputResultTypesTypes;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalLParen())) {

  inputOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    inputAttributesOperandsLoc = parser.getCurrentLocation();
    if (parseCreateOperationOpAttributes(parser, inputAttributesOperands, inputAttributeNamesAttr))
      return ::mlir::failure();
    result.addAttribute("inputAttributeNames", inputAttributeNamesAttr);
  }
  if (succeeded(parser.parseOptionalArrow())) {
  if (parser.parseLParen())
    return ::mlir::failure();

  inputResultTypesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputResultTypesOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputResultTypesTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(inputOperandsOperands.size()), static_cast<int32_t>(inputAttributesOperands.size()), static_cast<int32_t>(inputResultTypesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOperandsOperands, inputOperandsTypes, inputOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputAttributesOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputResultTypesOperands, inputResultTypesTypes, inputResultTypesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateOperationOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  if (!getInputOperands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << getInputOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getInputOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  printCreateOperationOpAttributes(_odsPrinter, *this, getInputAttributes(), getInputAttributeNamesAttr());
  if (!getInputResultTypes().empty()) {
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ' << "(";
  _odsPrinter << getInputResultTypes();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getInputResultTypes().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "name", "inputAttributeNames"});
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateOperationOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypeOp definitions
//===----------------------------------------------------------------------===//

CreateTypeOpAdaptor::CreateTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CreateTypeOpAdaptor::CreateTypeOpAdaptor(CreateTypeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CreateTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CreateTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CreateTypeOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("value").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type CreateTypeOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::LogicalResult CreateTypeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_value = odsAttrs.get("value");
    if (!tblgen_value)
      return emitError(loc, "'pdl_interp.create_type' op ""requires attribute 'value'");

    if (tblgen_value && !(((tblgen_value.isa<::mlir::TypeAttr>())) && ((tblgen_value.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
      return emitError(loc, "'pdl_interp.create_type' op ""attribute 'value' failed to satisfy constraint: any type attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateTypeOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::TypeAttr CreateTypeOp::getValueAttr() {
  return (*this)->getAttr(getValueAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type CreateTypeOp::getValue() {
  auto attr = getValueAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void CreateTypeOp::setValueAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeAttr type) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), type);
    
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::TypeAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type value) {
  odsState.addAttribute(getValueAttrName(odsState.name), ::mlir::TypeAttr::get(value));
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type value) {
  odsState.addAttribute(getValueAttrName(odsState.name), ::mlir::TypeAttr::get(value));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTypeOp::verifyInvariantsImpl() {
  {
    auto tblgen_value = (*this)->getAttr(getValueAttrName());
    if (!tblgen_value)
      return emitOpError("requires attribute 'value'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_value, "value")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getType<::mlir::NoneType>(), "value",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

void CreateTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypesOp definitions
//===----------------------------------------------------------------------===//

CreateTypesOpAdaptor::CreateTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CreateTypesOpAdaptor::CreateTypesOpAdaptor(CreateTypesOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CreateTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CreateTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CreateTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CreateTypesOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("value").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CreateTypesOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::LogicalResult CreateTypesOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_value = odsAttrs.get("value");
    if (!tblgen_value)
      return emitError(loc, "'pdl_interp.create_types' op ""requires attribute 'value'");

    if (tblgen_value && !(((tblgen_value.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_value.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
      return emitError(loc, "'pdl_interp.create_types' op ""attribute 'value' failed to satisfy constraint: type array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CreateTypesOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr CreateTypesOp::getValueAttr() {
  return (*this)->getAttr(getValueAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateTypesOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void CreateTypesOp::setValueAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayAttr type) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::TypeType>()), type);
    
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ArrayAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(result);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTypesOp::verifyInvariantsImpl() {
  {
    auto tblgen_value = (*this)->getAttr(getValueAttrName());
    if (!tblgen_value)
      return emitOpError("requires attribute 'value'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_value, "value")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getType<::mlir::NoneType>(), "value",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

void CreateTypesOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::EraseOp definitions
//===----------------------------------------------------------------------===//

EraseOpAdaptor::EraseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

EraseOpAdaptor::EraseOpAdaptor(EraseOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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

::mlir::Value EraseOpAdaptor::getInputOp() {
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

::mlir::Value EraseOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EraseOp::getInputOpMutable() {
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

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp) {
  odsState.addOperands(inputOp);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp) {
  odsState.addOperands(inputOp);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EraseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EraseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::EraseOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ExtractOp definitions
//===----------------------------------------------------------------------===//

ExtractOpAdaptor::ExtractOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ExtractOpAdaptor::ExtractOpAdaptor(ExtractOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ExtractOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractOpAdaptor::getRange() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExtractOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ExtractOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t ExtractOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_index = odsAttrs.get("index");
    if (!tblgen_index)
      return emitError(loc, "'pdl_interp.extract' op ""requires attribute 'index'");

    if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.extract' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractOp::getRange() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExtractOp::getRangeMutable() {
  auto range = getODSOperandIndexAndLength(0);
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
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr ExtractOp::getIndexAttr() {
  return (*this)->getAttr(getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t ExtractOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void ExtractOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value range, unsigned index) {
      build(odsBuilder, odsState,
            range.getType().cast<pdl::RangeType>().getElementType(),
            range, index);
    
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value range, ::mlir::IntegerAttr index) {
  odsState.addOperands(range);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value range, ::mlir::IntegerAttr index) {
  odsState.addOperands(range);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value range, uint32_t index) {
  odsState.addOperands(range);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value range, uint32_t index) {
  odsState.addOperands(range);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractOp::verifyInvariantsImpl() {
  {
    auto tblgen_index = (*this)->getAttr(getIndexAttrName());
    if (!tblgen_index)
      return emitOpError("requires attribute 'index'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(pdl::RangeType::get((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `range` is a PDL range whose element type matches type of `result`");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExtractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand rangeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rangeOperands(rangeRawOperands);  ::llvm::SMLoc rangeOperandsLoc;
  (void)rangeOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32), "index",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  rangeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rangeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!((type.isa<::mlir::pdl::PDLType>()))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be pdl type, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(rangeOperands, pdl::RangeType::get(resultTypes[0]), rangeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getRange();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void ExtractOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ExtractOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FinalizeOp definitions
//===----------------------------------------------------------------------===//

FinalizeOpAdaptor::FinalizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FinalizeOpAdaptor::FinalizeOpAdaptor(FinalizeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FinalizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FinalizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FinalizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FinalizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FinalizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FinalizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FinalizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FinalizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FinalizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FinalizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FinalizeOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult FinalizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FinalizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void FinalizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void FinalizeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FinalizeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ForEachOp definitions
//===----------------------------------------------------------------------===//

ForEachOpAdaptor::ForEachOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ForEachOpAdaptor::ForEachOpAdaptor(ForEachOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ForEachOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ForEachOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ForEachOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ForEachOpAdaptor::getValues() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ForEachOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange ForEachOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ForEachOpAdaptor::getRegion() {
  return *odsRegions[0];
}

::mlir::LogicalResult ForEachOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ForEachOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ForEachOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ForEachOp::getValues() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ForEachOp::getValuesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ForEachOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ForEachOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ForEachOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Block *ForEachOp::getSuccessor() {
  return (*this)->getSuccessor(0);
}

void ForEachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value values, ::mlir::Block *successor) {
  odsState.addOperands(values);
  (void)odsState.addRegion();
  odsState.addSuccessors(successor);
}

void ForEachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values, ::mlir::Block *successor) {
  odsState.addOperands(values);
  (void)odsState.addRegion();
  odsState.addSuccessors(successor);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForEachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ForEachOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLInterpOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForEachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ForEachOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FuncOp definitions
//===----------------------------------------------------------------------===//

FuncOpAdaptor::FuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FuncOpAdaptor::FuncOpAdaptor(FuncOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef FuncOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOpAdaptor::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("function_type").cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::FunctionType FuncOpAdaptor::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::RegionRange FuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FuncOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_sym_name = odsAttrs.get("sym_name");
    if (!tblgen_sym_name)
      return emitError(loc, "'pdl_interp.func' op ""requires attribute 'sym_name'");

    if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_function_type = odsAttrs.get("function_type");
    if (!tblgen_function_type)
      return emitError(loc, "'pdl_interp.func' op ""requires attribute 'function_type'");

    if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>()))))
      return emitError(loc, "'pdl_interp.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &FuncOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr FuncOp::getSymNameAttr() {
  return (*this)->getAttr(getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::getFunctionTypeAttr() {
  return (*this)->getAttr(getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

void FuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void FuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

::mlir::LogicalResult FuncOp::verifyInvariantsImpl() {
  {
    auto tblgen_sym_name = (*this)->getAttr(getSymNameAttrName());
    if (!tblgen_sym_name)
      return emitOpError("requires attribute 'sym_name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_sym_name, "sym_name")))
      return ::mlir::failure();
  }
  {
    auto tblgen_function_type = (*this)->getAttr(getFunctionTypeAttrName());
    if (!tblgen_function_type)
      return emitOpError("requires attribute 'function_type'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(*this, tblgen_function_type, "function_type")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLInterpOps1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FuncOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeOp definitions
//===----------------------------------------------------------------------===//

GetAttributeOpAdaptor::GetAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetAttributeOpAdaptor::GetAttributeOpAdaptor(GetAttributeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GetAttributeOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GetAttributeOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult GetAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_name = odsAttrs.get("name");
    if (!tblgen_name)
      return emitError(loc, "'pdl_interp.get_attribute' op ""requires attribute 'name'");

    if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.get_attribute' op ""attribute 'name' failed to satisfy constraint: string attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetAttributeOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeOp::getAttribute() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr GetAttributeOp::getNameAttr() {
  return (*this)->getAttr(getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GetAttributeOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void GetAttributeOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value inputOp, ::mlir::StringAttr name) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::StringAttr name) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value inputOp, ::llvm::StringRef name) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::llvm::StringRef name) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetAttributeOp::verifyInvariantsImpl() {
  {
    auto tblgen_name = (*this)->getAttr(getNameAttrName());
    if (!tblgen_name)
      return emitOpError("requires attribute 'name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
}

void GetAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeTypeOp definitions
//===----------------------------------------------------------------------===//

GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(GetAttributeTypeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetAttributeTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetAttributeTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetAttributeTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeTypeOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetAttributeTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetAttributeTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetAttributeTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetAttributeTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeTypeOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetAttributeTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetAttributeTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetAttributeTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetAttributeTypeOp::getResult() {
  return *getODSResults(0).begin();
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), value);
    
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetAttributeTypeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetAttributeTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetAttributeTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetAttributeTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetDefiningOpOp definitions
//===----------------------------------------------------------------------===//

GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(GetDefiningOpOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetDefiningOpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetDefiningOpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetDefiningOpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetDefiningOpOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetDefiningOpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetDefiningOpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetDefiningOpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetDefiningOpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetDefiningOpOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetDefiningOpOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetDefiningOpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetDefiningOpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetDefiningOpOp::getInputOp() {
  return *getODSResults(0).begin();
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type inputOp, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(inputOp);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetDefiningOpOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetDefiningOpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetDefiningOpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetDefiningOpOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetDefiningOpOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetDefiningOpOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandOp definitions
//===----------------------------------------------------------------------===//

GetOperandOpAdaptor::GetOperandOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetOperandOpAdaptor::GetOperandOpAdaptor(GetOperandOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetOperandOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetOperandOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetOperandOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetOperandOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetOperandOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult GetOperandOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_index = odsAttrs.get("index");
    if (!tblgen_index)
      return emitError(loc, "'pdl_interp.get_operand' op ""requires attribute 'index'");

    if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.get_operand' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetOperandOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetOperandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetOperandOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetOperandOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetOperandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandOp::getValue() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetOperandOp::getIndexAttr() {
  return (*this)->getAttr(getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t GetOperandOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetOperandOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetOperandOp::verifyInvariantsImpl() {
  {
    auto tblgen_index = (*this)->getAttr(getIndexAttrName());
    if (!tblgen_index)
      return emitOpError("requires attribute 'index'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetOperandOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetOperandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32), "index",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetOperandOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandsOp definitions
//===----------------------------------------------------------------------===//

GetOperandsOpAdaptor::GetOperandsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetOperandsOpAdaptor::GetOperandsOpAdaptor(GetOperandsOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetOperandsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetOperandsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetOperandsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandsOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetOperandsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandsOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint32_t> GetOperandsOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult GetOperandsOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_index = odsAttrs.get("index");
    if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.get_operands' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetOperandsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetOperandsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandsOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetOperandsOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetOperandsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetOperandsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetOperandsOp::getValue() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetOperandsOp::getIndexAttr() {
  return (*this)->getAttr(getIndexAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> GetOperandsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void GetOperandsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

::mlir::Attribute GetOperandsOp::removeIndexAttr() {
  return (*this)->removeAttr(getIndexAttrName());
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, Optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, inputOp,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  }
  odsState.addTypes(value);
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetOperandsOp::verifyInvariantsImpl() {
  {
    auto tblgen_index = (*this)->getAttr(getIndexAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetOperandsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetOperandsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

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

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("index")) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetOperandsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultOp definitions
//===----------------------------------------------------------------------===//

GetResultOpAdaptor::GetResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetResultOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetResultOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetResultOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_index = odsAttrs.get("index");
    if (!tblgen_index)
      return emitError(loc, "'pdl_interp.get_result' op ""requires attribute 'index'");

    if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.get_result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetResultOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetResultOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOp::getValue() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetResultOp::getIndexAttr() {
  return (*this)->getAttr(getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t GetResultOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getIndexAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetResultOp::verifyInvariantsImpl() {
  {
    auto tblgen_index = (*this)->getAttr(getIndexAttrName());
    if (!tblgen_index)
      return emitOpError("requires attribute 'index'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetResultOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32), "index",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetResultOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultsOp definitions
//===----------------------------------------------------------------------===//

GetResultsOpAdaptor::GetResultsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetResultsOpAdaptor::GetResultsOpAdaptor(GetResultsOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetResultsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetResultsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetResultsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultsOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetResultsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultsOpAdaptor::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("index").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint32_t> GetResultsOpAdaptor::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult GetResultsOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_index = odsAttrs.get("index");
    if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.get_results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetResultsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultsOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetResultsOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetResultsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultsOp::getValue() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetResultsOp::getIndexAttr() {
  return (*this)->getAttr(getIndexAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> GetResultsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void GetResultsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

::mlir::Attribute GetResultsOp::removeIndexAttr() {
  return (*this)->removeAttr(getIndexAttrName());
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, Optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, inputOp,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::ValueType>()), inputOp,
            IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  }
  odsState.addTypes(value);
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
  odsState.addAttribute(getIndexAttrName(odsState.name), index);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetResultsOp::verifyInvariantsImpl() {
  {
    auto tblgen_index = (*this)->getAttr(getIndexAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetResultsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetResultsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

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

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("index")) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"index"});
}

void GetResultsOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetUsersOp definitions
//===----------------------------------------------------------------------===//

GetUsersOpAdaptor::GetUsersOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetUsersOpAdaptor::GetUsersOpAdaptor(GetUsersOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetUsersOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetUsersOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetUsersOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetUsersOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetUsersOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetUsersOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetUsersOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetUsersOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetUsersOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetUsersOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetUsersOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetUsersOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetUsersOp::getOperations() {
  return *getODSResults(0).begin();
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::OperationType>()),
            value);
    
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type operations, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(operations);
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetUsersOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetUsersOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetUsersOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetUsersOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::OperationType>());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetUsersOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetUsersOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetUsersOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetValueTypeOp definitions
//===----------------------------------------------------------------------===//

GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(GetValueTypeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GetValueTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetValueTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetValueTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetValueTypeOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetValueTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetValueTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetValueTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetValueTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetValueTypeOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetValueTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetValueTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetValueTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetValueTypeOp::getResult() {
  return *getODSResults(0).begin();
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      Type valType = value.getType();
      Type typeType = odsBuilder.getType<pdl::TypeType>();
      build(odsBuilder, odsState,
            valType.isa<pdl::RangeType>() ? pdl::RangeType::get(typeType)
                                          : typeType,
            value);
    
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetValueTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetValueTypeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(getGetValueTypeOpValueType((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `value` type matches arity of `result`");
  return ::mlir::success();
}

::mlir::LogicalResult GetValueTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetValueTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!(((type.isa<::mlir::pdl::TypeType>())) || (((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>()))))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, getGetValueTypeOpValueType(resultTypes[0]), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetValueTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void GetValueTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetValueTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::InferredTypesOp definitions
//===----------------------------------------------------------------------===//

InferredTypesOpAdaptor::InferredTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

InferredTypesOpAdaptor::InferredTypesOpAdaptor(InferredTypesOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange InferredTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InferredTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InferredTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr InferredTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InferredTypesOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InferredTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InferredTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> InferredTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InferredTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InferredTypesOp::getResult() {
  return *getODSResults(0).begin();
}

void InferredTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::TypeType>()));
    
}

void InferredTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result) {
  odsState.addTypes(result);
}

void InferredTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InferredTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InferredTypesOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InferredTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult InferredTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void InferredTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::InferredTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::IsNotNullOp definitions
//===----------------------------------------------------------------------===//

IsNotNullOpAdaptor::IsNotNullOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

IsNotNullOpAdaptor::IsNotNullOpAdaptor(IsNotNullOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange IsNotNullOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IsNotNullOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IsNotNullOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsNotNullOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IsNotNullOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IsNotNullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IsNotNullOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IsNotNullOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IsNotNullOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IsNotNullOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IsNotNullOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IsNotNullOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *IsNotNullOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *IsNotNullOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsNotNullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IsNotNullOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult IsNotNullOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IsNotNullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IsNotNullOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void IsNotNullOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::IsNotNullOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::RecordMatchOp definitions
//===----------------------------------------------------------------------===//

RecordMatchOpAdaptor::RecordMatchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

RecordMatchOpAdaptor::RecordMatchOpAdaptor(RecordMatchOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange RecordMatchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RecordMatchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange RecordMatchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange RecordMatchOpAdaptor::getInputs() {
  return getODSOperands(0);
}

::mlir::ValueRange RecordMatchOpAdaptor::getMatchedOps() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr RecordMatchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr RecordMatchOpAdaptor::getRewriterAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::SymbolRefAttr attr = odsAttrs.get("rewriter").cast<::mlir::SymbolRefAttr>();
  return attr;
}

::mlir::SymbolRefAttr RecordMatchOpAdaptor::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOpAdaptor::getRootKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("rootKind").dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > RecordMatchOpAdaptor::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr RecordMatchOpAdaptor::getGeneratedOpsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("generatedOps").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > RecordMatchOpAdaptor::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr RecordMatchOpAdaptor::getBenefitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("benefit").cast<::mlir::IntegerAttr>();
  return attr;
}

uint16_t RecordMatchOpAdaptor::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult RecordMatchOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'pdl_interp.record_match' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'pdl_interp.record_match' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_rewriter = odsAttrs.get("rewriter");
    if (!tblgen_rewriter)
      return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'rewriter'");

    if (tblgen_rewriter && !((tblgen_rewriter.isa<::mlir::SymbolRefAttr>())))
      return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rewriter' failed to satisfy constraint: symbol reference attribute");
  }
  {
    auto tblgen_rootKind = odsAttrs.get("rootKind");
    if (tblgen_rootKind && !((tblgen_rootKind.isa<::mlir::StringAttr>())))
      return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_generatedOps = odsAttrs.get("generatedOps");
    if (tblgen_generatedOps && !(((tblgen_generatedOps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_generatedOps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
      return emitError(loc, "'pdl_interp.record_match' op ""attribute 'generatedOps' failed to satisfy constraint: string array attribute");
  }
  {
    auto tblgen_benefit = odsAttrs.get("benefit");
    if (!tblgen_benefit)
      return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'benefit'");

    if (tblgen_benefit && !((((tblgen_benefit.isa<::mlir::IntegerAttr>())) && ((tblgen_benefit.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!tblgen_benefit.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
      return emitError(loc, "'pdl_interp.record_match' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(getOperandSegmentSizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range RecordMatchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range RecordMatchOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range RecordMatchOp::getMatchedOps() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RecordMatchOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RecordMatchOp::getMatchedOpsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RecordMatchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *RecordMatchOp::getDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SymbolRefAttr RecordMatchOp::getRewriterAttr() {
  return (*this)->getAttr(getRewriterAttrName()).cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr RecordMatchOp::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOp::getRootKindAttr() {
  return (*this)->getAttr(getRootKindAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > RecordMatchOp::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr RecordMatchOp::getGeneratedOpsAttr() {
  return (*this)->getAttr(getGeneratedOpsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > RecordMatchOp::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr RecordMatchOp::getBenefitAttr() {
  return (*this)->getAttr(getBenefitAttrName()).cast<::mlir::IntegerAttr>();
}

uint16_t RecordMatchOp::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

void RecordMatchOp::setRewriterAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(getRewriterAttrName(), attr);
}

void RecordMatchOp::setRootKindAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getRootKindAttrName(), attr);
}

void RecordMatchOp::setGeneratedOpsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getGeneratedOpsAttrName(), attr);
}

void RecordMatchOp::setBenefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getBenefitAttrName(), attr);
}

::mlir::Attribute RecordMatchOp::removeRootKindAttr() {
  return (*this)->removeAttr(getRootKindAttrName());
}

::mlir::Attribute RecordMatchOp::removeGeneratedOpsAttr() {
  return (*this)->removeAttr(getGeneratedOpsAttrName());
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute(getRewriterAttrName(odsState.name), rewriter);
  if (rootKind) {
  odsState.addAttribute(getRootKindAttrName(odsState.name), rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute(getGeneratedOpsAttrName(odsState.name), generatedOps);
  }
  odsState.addAttribute(getBenefitAttrName(odsState.name), benefit);
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute(getRewriterAttrName(odsState.name), rewriter);
  if (rootKind) {
  odsState.addAttribute(getRootKindAttrName(odsState.name), rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute(getGeneratedOpsAttrName(odsState.name), generatedOps);
  }
  odsState.addAttribute(getBenefitAttrName(odsState.name), benefit);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute(getRewriterAttrName(odsState.name), rewriter);
  if (rootKind) {
  odsState.addAttribute(getRootKindAttrName(odsState.name), rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute(getGeneratedOpsAttrName(odsState.name), generatedOps);
  }
  odsState.addAttribute(getBenefitAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.addAttribute(getRewriterAttrName(odsState.name), rewriter);
  if (rootKind) {
  odsState.addAttribute(getRootKindAttrName(odsState.name), rootKind);
  }
  if (generatedOps) {
  odsState.addAttribute(getGeneratedOpsAttrName(odsState.name), generatedOps);
  }
  odsState.addAttribute(getBenefitAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit));
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RecordMatchOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(getOperandSegmentSizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_rewriter = (*this)->getAttr(getRewriterAttrName());
    if (!tblgen_rewriter)
      return emitOpError("requires attribute 'rewriter'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_rewriter, "rewriter")))
      return ::mlir::failure();
  }
  {
    auto tblgen_rootKind = (*this)->getAttr(getRootKindAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_rootKind, "rootKind")))
      return ::mlir::failure();
  }
  {
    auto tblgen_generatedOps = (*this)->getAttr(getGeneratedOpsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_generatedOps, "generatedOps")))
      return ::mlir::failure();
  }
  {
    auto tblgen_benefit = (*this)->getAttr(getBenefitAttrName());
    if (!tblgen_benefit)
      return emitOpError("requires attribute 'benefit'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps9(*this, tblgen_benefit, "benefit")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult RecordMatchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult RecordMatchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr rewriterAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::ArrayAttr generatedOpsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> matchedOpsOperands;
  ::llvm::SMLoc matchedOpsOperandsLoc;
  (void)matchedOpsOperandsLoc;
  ::mlir::StringAttr rootKindAttr;
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseCustomAttributeWithFallback(rewriterAttr, parser.getBuilder().getType<::mlir::NoneType>(), "rewriter",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalLParen())) {

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
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
  if (parser.parseComma())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("generatedOps"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(generatedOpsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "generatedOps",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("loc"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  matchedOpsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(matchedOpsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("root"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(rootKindAttr, parser.getBuilder().getType<::mlir::NoneType>(), "rootKind",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(inputsOperands.size()), static_cast<int32_t>(matchedOpsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matchedOpsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RecordMatchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getRewriterAttr());
  if (!getInputs().empty()) {
  _odsPrinter << "(";
  _odsPrinter << getInputs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getInputs().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "benefit";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getBenefitAttr());
  _odsPrinter << ")";
  _odsPrinter << ",";
  if ((*this)->getAttr("generatedOps")) {
  _odsPrinter << ' ' << "generatedOps";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getGeneratedOpsAttr());
  _odsPrinter << ")";
  _odsPrinter << ",";
  }
  _odsPrinter << ' ' << "loc";
  _odsPrinter << "(";
  _odsPrinter << "[";
  _odsPrinter << getMatchedOps();
  _odsPrinter << "]";
  _odsPrinter << ")";
  if ((*this)->getAttr("rootKind")) {
  _odsPrinter << ",";
  _odsPrinter << ' ' << "root";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getRootKindAttr());
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "rewriter", "benefit", "generatedOps", "rootKind"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::RecordMatchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ReplaceOp definitions
//===----------------------------------------------------------------------===//

ReplaceOpAdaptor::ReplaceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ReplaceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReplaceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReplaceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReplaceOpAdaptor::getReplValues() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ReplaceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReplaceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReplaceOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ReplaceOp::getReplValues() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ReplaceOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReplaceOp::getReplValuesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
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

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::ValueRange replValues) {
  odsState.addOperands(inputOp);
  odsState.addOperands(replValues);
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::ValueRange replValues) {
  odsState.addOperands(inputOp);
  odsState.addOperands(replValues);
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
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReplaceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> replValuesTypes;

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (!replValuesOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(replValuesTypes))
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "with";
  _odsPrinter << ' ';
  _odsPrinter << "(";
  if (!getReplValues().empty()) {
  _odsPrinter << getReplValues();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getReplValues().getTypes();
  }
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ReplaceOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchAttributeOp definitions
//===----------------------------------------------------------------------===//

SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(SwitchAttributeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SwitchAttributeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchAttributeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchAttributeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchAttributeOpAdaptor::getAttribute() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchAttributeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchAttributeOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchAttributeOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

::mlir::LogicalResult SwitchAttributeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_caseValues = odsAttrs.get("caseValues");
    if (!tblgen_caseValues)
      return emitError(loc, "'pdl_interp.switch_attribute' op ""requires attribute 'caseValues'");

    if (tblgen_caseValues && !((tblgen_caseValues.isa<::mlir::ArrayAttr>())))
      return emitError(loc, "'pdl_interp.switch_attribute' op ""attribute 'caseValues' failed to satisfy constraint: array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchAttributeOp::getAttribute() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchAttributeOp::getAttributeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchAttributeOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchAttributeOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchAttributeOp::getCaseValuesAttr() {
  return (*this)->getAttr(getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchAttributeOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchAttributeOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value attribute, ArrayRef<Attribute> caseValues, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, attribute, odsBuilder.getArrayAttr(caseValues),
          defaultDest, dests);
  
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchAttributeOp::verifyInvariantsImpl() {
  {
    auto tblgen_caseValues = (*this)->getAttr(getCaseValuesAttrName());
    if (!tblgen_caseValues)
      return emitOpError("requires attribute 'caseValues'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(*this, tblgen_caseValues, "caseValues")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchAttributeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand attributeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> attributeOperands(attributeRawOperands);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchAttributeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperandCountOp definitions
//===----------------------------------------------------------------------===//

SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(SwitchOperandCountOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SwitchOperandCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchOperandCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchOperandCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperandCountOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchOperandCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("caseValues").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

::mlir::LogicalResult SwitchOperandCountOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_caseValues = odsAttrs.get("caseValues");
    if (!tblgen_caseValues)
      return emitError(loc, "'pdl_interp.switch_operand_count' op ""requires attribute 'caseValues'");

    if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_caseValues.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
      return emitError(loc, "'pdl_interp.switch_operand_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOperandCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchOperandCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperandCountOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchOperandCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchOperandCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOperandCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOperandCountOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOperandCountOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::getCaseValuesAttr() {
  return (*this)->getAttr(getCaseValuesAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchOperandCountOp::setCaseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<int32_t> counts, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, inputOp, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOperandCountOp::verifyInvariantsImpl() {
  {
    auto tblgen_caseValues = (*this)->getAttr(getCaseValuesAttrName());
    if (!tblgen_caseValues)
      return emitOpError("requires attribute 'caseValues'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(*this, tblgen_caseValues, "caseValues")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOperandCountOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, ::mlir::Type{}, "caseValues",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperandCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperandCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperationNameOp definitions
//===----------------------------------------------------------------------===//

SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(SwitchOperationNameOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SwitchOperationNameOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchOperationNameOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchOperationNameOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperationNameOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchOperationNameOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchOperationNameOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchOperationNameOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

::mlir::LogicalResult SwitchOperationNameOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_caseValues = odsAttrs.get("caseValues");
    if (!tblgen_caseValues)
      return emitError(loc, "'pdl_interp.switch_operation_name' op ""requires attribute 'caseValues'");

    if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
      return emitError(loc, "'pdl_interp.switch_operation_name' op ""attribute 'caseValues' failed to satisfy constraint: string array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOperationNameOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchOperationNameOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOperationNameOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchOperationNameOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchOperationNameOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOperationNameOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOperationNameOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOperationNameOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchOperationNameOp::getCaseValuesAttr() {
  return (*this)->getAttr(getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchOperationNameOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchOperationNameOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<OperationName> names, Block *defaultDest, BlockRange dests) {
      auto stringNames = llvm::to_vector<8>(llvm::map_range(names,
          [](OperationName name) { return name.getStringRef(); }));
      build(odsBuilder, odsState, inputOp, odsBuilder.getStrArrayAttr(stringNames),
            defaultDest, dests);
    
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOperationNameOp::verifyInvariantsImpl() {
  {
    auto tblgen_caseValues = (*this)->getAttr(getCaseValuesAttrName());
    if (!tblgen_caseValues)
      return emitOpError("requires attribute 'caseValues'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_caseValues, "caseValues")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOperationNameOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperationNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperationNameOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchResultCountOp definitions
//===----------------------------------------------------------------------===//

SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(SwitchResultCountOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SwitchResultCountOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchResultCountOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchResultCountOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchResultCountOpAdaptor::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchResultCountOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::DenseIntElementsAttr attr = odsAttrs.get("caseValues").cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

::mlir::LogicalResult SwitchResultCountOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_caseValues = odsAttrs.get("caseValues");
    if (!tblgen_caseValues)
      return emitError(loc, "'pdl_interp.switch_result_count' op ""requires attribute 'caseValues'");

    if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_caseValues.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
      return emitError(loc, "'pdl_interp.switch_result_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchResultCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchResultCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchResultCountOp::getInputOp() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchResultCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchResultCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchResultCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchResultCountOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchResultCountOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::getCaseValuesAttr() {
  return (*this)->getAttr(getCaseValuesAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchResultCountOp::setCaseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<int32_t> counts, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, inputOp, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchResultCountOp::verifyInvariantsImpl() {
  {
    auto tblgen_caseValues = (*this)->getAttr(getCaseValuesAttrName());
    if (!tblgen_caseValues)
      return emitOpError("requires attribute 'caseValues'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(*this, tblgen_caseValues, "caseValues")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchResultCountOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, ::mlir::Type{}, "caseValues",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchResultCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchResultCountOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypeOp definitions
//===----------------------------------------------------------------------===//

SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(SwitchTypeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SwitchTypeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchTypeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchTypeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypeOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchTypeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypeOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchTypeOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

::mlir::LogicalResult SwitchTypeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_caseValues = odsAttrs.get("caseValues");
    if (!tblgen_caseValues)
      return emitError(loc, "'pdl_interp.switch_type' op ""requires attribute 'caseValues'");

    if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
      return emitError(loc, "'pdl_interp.switch_type' op ""attribute 'caseValues' failed to satisfy constraint: type array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypeOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchTypeOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchTypeOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchTypeOp::getCaseValuesAttr() {
  return (*this)->getAttr(getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchTypeOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchTypeOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block *defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchTypeOp::verifyInvariantsImpl() {
  {
    auto tblgen_caseValues = (*this)->getAttr(getCaseValuesAttrName());
    if (!tblgen_caseValues)
      return emitOpError("requires attribute 'caseValues'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_caseValues, "caseValues")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchTypeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypesOp definitions
//===----------------------------------------------------------------------===//

SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(SwitchTypesOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SwitchTypesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchTypesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SwitchTypesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypesOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SwitchTypesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypesOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("caseValues").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchTypesOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

::mlir::LogicalResult SwitchTypesOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_caseValues = odsAttrs.get("caseValues");
    if (!tblgen_caseValues)
      return emitError(loc, "'pdl_interp.switch_types' op ""requires attribute 'caseValues'");

    if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))); }))))
      return emitError(loc, "'pdl_interp.switch_types' op ""attribute 'caseValues' failed to satisfy constraint: type-array array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchTypesOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SwitchTypesOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchTypesOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchTypesOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ArrayAttr SwitchTypesOp::getCaseValuesAttr() {
  return (*this)->getAttr(getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchTypesOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchTypesOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block *defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), caseValues);
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchTypesOp::verifyInvariantsImpl() {
  {
    auto tblgen_caseValues = (*this)->getAttr(getCaseValuesAttrName());
    if (!tblgen_caseValues)
      return emitOpError("requires attribute 'caseValues'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(*this, tblgen_caseValues, "caseValues")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchTypesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "caseValues",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.hasValue()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"caseValues"});
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypesOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypesOp)


#endif  // GET_OP_CLASSES

