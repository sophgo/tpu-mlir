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
::mlir::pdl_interp::CreateRangeOp,
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
  if (!(((type.isa<::mlir::pdl::RangeType>())) && (((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::TypeType>())) || ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::ValueType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` or PDL handle for an `mlir::Value` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::pdl::RangeType>())) && ((type.cast<::mlir::pdl::RangeType>().getElementType().isa<::mlir::pdl::PDLType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of pdl type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::ValueType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps10(
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
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))); })))) {
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
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Array of dictionary attributes";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::SymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!attr.cast<::mlir::IntegerAttr>().getValue().isNegative())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))); }))); })))) {
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

namespace detail {
ApplyConstraintOpGenericAdaptorBase::ApplyConstraintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.apply_constraint", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ApplyConstraintOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ApplyConstraintOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyConstraintOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyConstraintOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyConstraintOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(ApplyConstraintOp op) : ApplyConstraintOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ApplyConstraintOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.apply_constraint' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == ApplyConstraintOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyConstraintOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyConstraintOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyConstraintOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
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
    if (firstSucc.has_value()) {
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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

namespace detail {
ApplyRewriteOpGenericAdaptorBase::ApplyRewriteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.apply_rewrite", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ApplyRewriteOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ApplyRewriteOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyRewriteOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyRewriteOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyRewriteOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(ApplyRewriteOp op) : ApplyRewriteOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ApplyRewriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.apply_rewrite' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == ApplyRewriteOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.apply_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyRewriteOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyRewriteOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyRewriteOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyRewriteOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::AreEqualOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AreEqualOpGenericAdaptorBase::AreEqualOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.are_equal", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AreEqualOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AreEqualOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AreEqualOpAdaptor::AreEqualOpAdaptor(AreEqualOp op) : AreEqualOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::PDLType> AreEqualOp::getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::pdl::PDLType> AreEqualOp::getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(1).begin());
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(allOperands, ::llvm::concat<const ::mlir::Type>(::llvm::ArrayRef<::mlir::Type>(lhsTypes), ::llvm::ArrayRef<::mlir::Type>(lhsTypes[0])), allOperandLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void AreEqualOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::AreEqualOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::BranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BranchOpGenericAdaptorBase::BranchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.branch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BranchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BranchOpAdaptor::BranchOpAdaptor(BranchOp op) : BranchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void BranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::BranchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckAttributeOpGenericAdaptorBase::CheckAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CheckAttributeOpGenericAdaptorBase::getConstantValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CheckAttributeOp::getConstantValueAttrName(*odsOpName)).cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute CheckAttributeOpGenericAdaptorBase::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

} // namespace detail
CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(CheckAttributeOp op) : CheckAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CheckAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_constantValue;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.check_attribute' op ""requires attribute 'constantValue'");
    if (namedAttrIt->getName() == CheckAttributeOp::getConstantValueAttrName(*odsOpName)) {
      tblgen_constantValue = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_constantValue && !((true)))
    return emitError(loc, "'pdl_interp.check_attribute' op ""attribute 'constantValue' failed to satisfy constraint: any attribute");
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

::mlir::TypedValue<::mlir::pdl::AttributeType> CheckAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getConstantValueAttrName()).cast<::mlir::Attribute>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_constantValue;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'constantValue'");
    if (namedAttrIt->getName() == getConstantValueAttrName()) {
      tblgen_constantValue = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_constantValue, "constantValue")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, attributeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getConstantValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("constantValue");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperandCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckOperandCountOpGenericAdaptorBase::CheckOperandCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_operand_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckOperandCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckOperandCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckOperandCountOpGenericAdaptorBase::getCountAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CheckOperandCountOp::getCountAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t CheckOperandCountOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOpGenericAdaptorBase::getCompareAtLeastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CheckOperandCountOp::getCompareAtLeastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CheckOperandCountOpGenericAdaptorBase::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(CheckOperandCountOp op) : CheckOperandCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CheckOperandCountOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_count;
  ::mlir::Attribute tblgen_compareAtLeast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.check_operand_count' op ""requires attribute 'count'");
    if (namedAttrIt->getName() == CheckOperandCountOp::getCountAttrName(*odsOpName)) {
      tblgen_count = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CheckOperandCountOp::getCompareAtLeastAttrName(*odsOpName)) {
      tblgen_compareAtLeast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_count && !((((tblgen_count.isa<::mlir::IntegerAttr>())) && ((tblgen_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_count.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_compareAtLeast && !((tblgen_compareAtLeast.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> CheckOperandCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCountAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t CheckOperandCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOp::getCompareAtLeastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCompareAtLeastAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckOperandCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CheckOperandCountOp::setCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCountAttrName(), attr);
}

void CheckOperandCountOp::setCount(uint32_t attrValue) {
  (*this)->setAttr(getCountAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CheckOperandCountOp::setCompareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCompareAtLeastAttrName(), attr);
}

void CheckOperandCountOp::setCompareAtLeast(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCompareAtLeastAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getCompareAtLeastAttrName());
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
    odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
    odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr));
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_count;
  ::mlir::Attribute tblgen_compareAtLeast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'count'");
    if (namedAttrIt->getName() == getCountAttrName()) {
      tblgen_count = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCompareAtLeastAttrName()) {
      tblgen_compareAtLeast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_count, "count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_compareAtLeast, "compareAtLeast")))
    return ::mlir::failure();
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
  if (::mlir::succeeded(parser.parseOptionalKeyword("at_least"))) {
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("compareAtLeast");
  elidedAttrs.push_back("count");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCompareAtLeastAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("compareAtLeast");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperandCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperationNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckOperationNameOpGenericAdaptorBase::CheckOperationNameOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_operation_name", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckOperationNameOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckOperationNameOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CheckOperationNameOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CheckOperationNameOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CheckOperationNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(CheckOperationNameOp op) : CheckOperationNameOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CheckOperationNameOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.check_operation_name' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == CheckOperationNameOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.check_operation_name' op ""attribute 'name' failed to satisfy constraint: string attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> CheckOperationNameOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CheckOperationNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CheckOperationNameOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CheckOperationNameOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperationNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckResultCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckResultCountOpGenericAdaptorBase::CheckResultCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_result_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckResultCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckResultCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckResultCountOpGenericAdaptorBase::getCountAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CheckResultCountOp::getCountAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t CheckResultCountOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOpGenericAdaptorBase::getCompareAtLeastAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CheckResultCountOp::getCompareAtLeastAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CheckResultCountOpGenericAdaptorBase::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(CheckResultCountOp op) : CheckResultCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CheckResultCountOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_count;
  ::mlir::Attribute tblgen_compareAtLeast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.check_result_count' op ""requires attribute 'count'");
    if (namedAttrIt->getName() == CheckResultCountOp::getCountAttrName(*odsOpName)) {
      tblgen_count = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CheckResultCountOp::getCompareAtLeastAttrName(*odsOpName)) {
      tblgen_compareAtLeast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_count && !((((tblgen_count.isa<::mlir::IntegerAttr>())) && ((tblgen_count.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_count.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_compareAtLeast && !((tblgen_compareAtLeast.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> CheckResultCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCountAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t CheckResultCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOp::getCompareAtLeastAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCompareAtLeastAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckResultCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CheckResultCountOp::setCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCountAttrName(), attr);
}

void CheckResultCountOp::setCount(uint32_t attrValue) {
  (*this)->setAttr(getCountAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CheckResultCountOp::setCompareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCompareAtLeastAttrName(), attr);
}

void CheckResultCountOp::setCompareAtLeast(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCompareAtLeastAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getCompareAtLeastAttrName());
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
    odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.addAttribute(getCountAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count));
  if (compareAtLeast) {
    odsState.addAttribute(getCompareAtLeastAttrName(odsState.name), ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr));
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_count;
  ::mlir::Attribute tblgen_compareAtLeast;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'count'");
    if (namedAttrIt->getName() == getCountAttrName()) {
      tblgen_count = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCompareAtLeastAttrName()) {
      tblgen_compareAtLeast = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_count, "count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_compareAtLeast, "compareAtLeast")))
    return ::mlir::failure();
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
  if (::mlir::succeeded(parser.parseOptionalKeyword("at_least"))) {
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("compareAtLeast");
  elidedAttrs.push_back("count");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCompareAtLeastAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("compareAtLeast");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckResultCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckTypeOpGenericAdaptorBase::CheckTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CheckTypeOpGenericAdaptorBase::getTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CheckTypeOp::getTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type CheckTypeOpGenericAdaptorBase::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

} // namespace detail
CheckTypeOpAdaptor::CheckTypeOpAdaptor(CheckTypeOp op) : CheckTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CheckTypeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.check_type' op ""requires attribute 'type'");
    if (namedAttrIt->getName() == CheckTypeOp::getTypeAttrName(*odsOpName)) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'pdl_interp.check_type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
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

::mlir::TypedValue<::mlir::pdl::TypeType> CheckTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type CheckTypeOp::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void CheckTypeOp::setTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTypeAttrName(), attr);
}

void CheckTypeOp::setType(::mlir::Type attrValue) {
  (*this)->setAttr(getTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'type'");
    if (namedAttrIt->getName() == getTypeAttrName()) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_type, "type")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("type");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckTypesOpGenericAdaptorBase::CheckTypesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_types", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckTypesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckTypesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CheckTypesOpGenericAdaptorBase::getTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CheckTypesOp::getTypesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CheckTypesOpGenericAdaptorBase::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

} // namespace detail
CheckTypesOpAdaptor::CheckTypesOpAdaptor(CheckTypesOp op) : CheckTypesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CheckTypesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.check_types' op ""requires attribute 'types'");
    if (namedAttrIt->getName() == CheckTypesOp::getTypesAttrName(*odsOpName)) {
      tblgen_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_types && !(((tblgen_types.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_types.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))); }))))
    return emitError(loc, "'pdl_interp.check_types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
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

::mlir::TypedValue<::mlir::pdl::RangeType> CheckTypesOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTypesAttrName()).cast<::mlir::ArrayAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_types;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'types'");
    if (namedAttrIt->getName() == getTypesAttrName()) {
      tblgen_types = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_types, "types")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "are";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("types");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ContinueOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ContinueOpGenericAdaptorBase::ContinueOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.continue", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ContinueOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ContinueOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ContinueOpAdaptor::ContinueOpAdaptor(ContinueOp op) : ContinueOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ContinueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ContinueOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateAttributeOpGenericAdaptorBase::CreateAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CreateAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CreateAttributeOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateAttributeOp::getValueAttrName(*odsOpName)).cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute CreateAttributeOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(CreateAttributeOp op) : CreateAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CreateAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.create_attribute' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == CreateAttributeOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((true)))
    return emitError(loc, "'pdl_interp.create_attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
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

::mlir::TypedValue<::mlir::pdl::AttributeType> CreateAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSResults(0).begin());
}

::mlir::Attribute CreateAttributeOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::Attribute>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_value, "value")))
    return ::mlir::failure();
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

void CreateAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateOperationOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateOperationOpGenericAdaptorBase::CreateOperationOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_operation", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateOperationOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, CreateOperationOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CreateOperationOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CreateOperationOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, CreateOperationOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CreateOperationOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOpGenericAdaptorBase::getInputAttributeNamesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, CreateOperationOp::getInputAttributeNamesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CreateOperationOpGenericAdaptorBase::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

::mlir::UnitAttr CreateOperationOpGenericAdaptorBase::getInferredResultTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, CreateOperationOp::getInferredResultTypesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CreateOperationOpGenericAdaptorBase::getInferredResultTypes() {
  auto attr = getInferredResultTypesAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CreateOperationOpAdaptor::CreateOperationOpAdaptor(CreateOperationOp op) : CreateOperationOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CreateOperationOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inputAttributeNames;
  ::mlir::Attribute tblgen_inferredResultTypes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'inputAttributeNames'");
    if (namedAttrIt->getName() == CreateOperationOp::getInputAttributeNamesAttrName(*odsOpName)) {
      tblgen_inputAttributeNames = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CreateOperationOp::getInferredResultTypesAttrName(*odsOpName)) {
      tblgen_inferredResultTypes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == CreateOperationOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CreateOperationOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'pdl_interp.create_operation' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_inputAttributeNames && !(((tblgen_inputAttributeNames.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_inputAttributeNames.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inputAttributeNames' failed to satisfy constraint: string array attribute");

  if (tblgen_inferredResultTypes && !((tblgen_inferredResultTypes.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inferredResultTypes' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
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
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputAttributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputResultTypesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
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

::mlir::TypedValue<::mlir::pdl::OperationType> CreateOperationOp::getResultOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSResults(0).begin());
}

::mlir::StringAttr CreateOperationOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CreateOperationOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNamesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getInputAttributeNamesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

::mlir::UnitAttr CreateOperationOp::getInferredResultTypesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getInferredResultTypesAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CreateOperationOp::getInferredResultTypes() {
  auto attr = getInferredResultTypesAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CreateOperationOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CreateOperationOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateOperationOp::setInputAttributeNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInputAttributeNamesAttrName(), attr);
}

void CreateOperationOp::setInferredResultTypesAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInferredResultTypesAttrName(), attr);
}

void CreateOperationOp::setInferredResultTypes(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInferredResultTypesAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInferredResultTypesAttrName());
}

::mlir::Attribute CreateOperationOp::removeInferredResultTypesAttr() {
  return (*this)->removeAttr(getInferredResultTypesAttrName());
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange types, bool inferredResultTypes, ValueRange operands, ValueRange attributes, ArrayAttr attributeNames) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types, inferredResultTypes);
    
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/::mlir::UnitAttr inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  if (inferredResultTypes) {
    odsState.addAttribute(getInferredResultTypesAttrName(odsState.name), inferredResultTypes);
  }
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/::mlir::UnitAttr inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  if (inferredResultTypes) {
    odsState.addAttribute(getInferredResultTypesAttrName(odsState.name), inferredResultTypes);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/bool inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  if (inferredResultTypes) {
    odsState.addAttribute(getInferredResultTypesAttrName(odsState.name), ((inferredResultTypes) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/bool inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getInputAttributeNamesAttrName(odsState.name), inputAttributeNames);
  if (inferredResultTypes) {
    odsState.addAttribute(getInferredResultTypesAttrName(odsState.name), ((inferredResultTypes) ? odsBuilder.getUnitAttr() : nullptr));
  }
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inputAttributeNames;
  ::mlir::Attribute tblgen_inferredResultTypes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'inputAttributeNames'");
    if (namedAttrIt->getName() == getInputAttributeNamesAttrName()) {
      tblgen_inputAttributeNames = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getInferredResultTypesAttrName()) {
      tblgen_inferredResultTypes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_inputAttributeNames, "inputAttributeNames")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_inferredResultTypes, "inferredResultTypes")))
    return ::mlir::failure();
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
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateOperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputOperandsOperands;
  ::llvm::SMLoc inputOperandsOperandsLoc;
  (void)inputOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputAttributesOperands;
  ::llvm::SMLoc inputAttributesOperandsLoc;
  (void)inputAttributesOperandsLoc;
  ::mlir::ArrayAttr inputAttributeNamesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputResultTypesOperands;
  ::llvm::SMLoc inputResultTypesOperandsLoc;
  (void)inputResultTypesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputResultTypesTypes;
  ::mlir::UnitAttr inferredResultTypesAttr;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

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
  {
    inputResultTypesOperandsLoc = parser.getCurrentLocation();
    if (parseCreateOperationOpResults(parser, inputResultTypesOperands, inputResultTypesTypes, inferredResultTypesAttr))
      return ::mlir::failure();
    if (inferredResultTypesAttr)
      result.addAttribute("inferredResultTypes", inferredResultTypesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(inputOperandsOperands.size()), static_cast<int32_t>(inputAttributesOperands.size()), static_cast<int32_t>(inputResultTypesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOperandsOperands, inputOperandsTypes, inputOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputAttributesOperands, odsBuildableType1, inputAttributesOperandsLoc, result.operands))
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
  printCreateOperationOpAttributes(_odsPrinter, *this, getInputAttributes(), getInputAttributeNamesAttr());
  _odsPrinter << ' ';
  printCreateOperationOpResults(_odsPrinter, *this, getInputResultTypes(), getInputResultTypes().getTypes(), getInferredResultTypesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("name");
  elidedAttrs.push_back("inputAttributeNames");
  elidedAttrs.push_back("inferredResultTypes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInferredResultTypesAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inferredResultTypes");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateOperationOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateRangeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateRangeOpGenericAdaptorBase::CreateRangeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_range", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateRangeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr CreateRangeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CreateRangeOpAdaptor::CreateRangeOpAdaptor(CreateRangeOp op) : CreateRangeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CreateRangeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateRangeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CreateRangeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateRangeOp::getArguments() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CreateRangeOp::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CreateRangeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateRangeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> CreateRangeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSResults(0).begin());
}

void CreateRangeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addTypes(result);
}

void CreateRangeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateRangeOp::verifyInvariantsImpl() {
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateRangeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateRangeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void CreateRangeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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

void CreateRangeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateRangeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateTypeOpGenericAdaptorBase::CreateTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CreateTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CreateTypeOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateTypeOp::getValueAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type CreateTypeOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr.getValue().cast<::mlir::Type>();
}

} // namespace detail
CreateTypeOpAdaptor::CreateTypeOpAdaptor(CreateTypeOp op) : CreateTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CreateTypeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.create_type' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == CreateTypeOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !(((tblgen_value.isa<::mlir::TypeAttr>())) && ((tblgen_value.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))))
    return emitError(loc, "'pdl_interp.create_type' op ""attribute 'value' failed to satisfy constraint: any type attribute");
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

::mlir::TypedValue<::mlir::pdl::TypeType> CreateTypeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSResults(0).begin());
}

::mlir::TypeAttr CreateTypeOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type CreateTypeOp::getValue() {
  auto attr = getValueAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void CreateTypeOp::setValueAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateTypeOp::setValue(::mlir::Type attrValue) {
  (*this)->setAttr(getValueAttrName(), ::mlir::TypeAttr::get(attrValue));
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_value, "value")))
    return ::mlir::failure();
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateTypesOpGenericAdaptorBase::CreateTypesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_types", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateTypesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CreateTypesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CreateTypesOpGenericAdaptorBase::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CreateTypesOp::getValueAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CreateTypesOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
CreateTypesOpAdaptor::CreateTypesOpAdaptor(CreateTypesOp op) : CreateTypesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CreateTypesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.create_types' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == CreateTypesOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !(((tblgen_value.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_value.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))); }))))
    return emitError(loc, "'pdl_interp.create_types' op ""attribute 'value' failed to satisfy constraint: type array attribute");
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

::mlir::TypedValue<::mlir::pdl::RangeType> CreateTypesOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSResults(0).begin());
}

::mlir::ArrayAttr CreateTypesOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::ArrayAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_value, "value")))
    return ::mlir::failure();
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::EraseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EraseOpGenericAdaptorBase::EraseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.erase", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EraseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr EraseOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
EraseOpAdaptor::EraseOpAdaptor(EraseOp op) : EraseOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::OperationType> EraseOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::EraseOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ExtractOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractOpGenericAdaptorBase::ExtractOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.extract", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExtractOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ExtractOpGenericAdaptorBase::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExtractOp::getIndexAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t ExtractOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ExtractOpAdaptor::ExtractOpAdaptor(ExtractOp op) : ExtractOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.extract' op ""requires attribute 'index'");
    if (namedAttrIt->getName() == ExtractOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.extract' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
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

::mlir::TypedValue<::mlir::pdl::RangeType> ExtractOp::getRange() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::PDLType> ExtractOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr ExtractOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t ExtractOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void ExtractOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void ExtractOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "operand", index++)))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ExtractOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ExtractOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FinalizeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FinalizeOpGenericAdaptorBase::FinalizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.finalize", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FinalizeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FinalizeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FinalizeOpAdaptor::FinalizeOpAdaptor(FinalizeOp op) : FinalizeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void FinalizeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FinalizeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ForEachOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ForEachOpGenericAdaptorBase::ForEachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.foreach", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ForEachOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ForEachOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &ForEachOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ForEachOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ForEachOpAdaptor::ForEachOpAdaptor(ForEachOp op) : ForEachOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::RangeType> ForEachOp::getValues() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
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

namespace detail {
FuncOpGenericAdaptorBase::FuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.func", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FuncOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FuncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, FuncOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef FuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOpGenericAdaptorBase::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, FuncOp::getFunctionTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::FunctionType FuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getArgAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, FuncOp::getArgAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getResAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, FuncOp::getResAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::Region &FuncOpGenericAdaptorBase::getBody() {
  return *odsRegions[0];
}

::mlir::RegionRange FuncOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
FuncOpAdaptor::FuncOpAdaptor(FuncOp op) : FuncOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_arg_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.func' op ""requires attribute 'function_type'");
    if (namedAttrIt->getName() == FuncOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == FuncOp::getArgAttrsAttrName(*odsOpName)) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_res_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.func' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == FuncOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == FuncOp::getResAttrsAttrName(*odsOpName)) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>()))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_arg_attrs && !(((tblgen_arg_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_arg_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'arg_attrs' failed to satisfy constraint: Array of dictionary attributes");

  if (tblgen_res_attrs && !(((tblgen_res_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_res_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'res_attrs' failed to satisfy constraint: Array of dictionary attributes");
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::ArrayAttr FuncOp::getArgAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getArgAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOp::getResAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getResAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void FuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void FuncOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void FuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void FuncOp::setFunctionType(::mlir::FunctionType attrValue) {
  (*this)->setAttr(getFunctionTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void FuncOp::setArgAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgAttrsAttrName(), attr);
}

void FuncOp::setResAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getResAttrsAttrName(), attr);
}

::mlir::Attribute FuncOp::removeArgAttrsAttr() {
  return (*this)->removeAttr(getArgAttrsAttrName());
}

::mlir::Attribute FuncOp::removeResAttrsAttr() {
  return (*this)->removeAttr(getResAttrsAttrName());
}

::mlir::LogicalResult FuncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_arg_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getArgAttrsAttrName()) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_res_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getResAttrsAttrName()) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
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

namespace detail {
GetAttributeOpGenericAdaptorBase::GetAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GetAttributeOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetAttributeOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GetAttributeOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
GetAttributeOpAdaptor::GetAttributeOpAdaptor(GetAttributeOp op) : GetAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.get_attribute' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == GetAttributeOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.get_attribute' op ""attribute 'name' failed to satisfy constraint: string attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetAttributeOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::AttributeType> GetAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSResults(0).begin());
}

::mlir::StringAttr GetAttributeOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GetAttributeOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void GetAttributeOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void GetAttributeOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetAttributeTypeOpGenericAdaptorBase::GetAttributeTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_attribute_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetAttributeTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetAttributeTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(GetAttributeTypeOp op) : GetAttributeTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::AttributeType> GetAttributeTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::TypeType> GetAttributeTypeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSResults(0).begin());
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
  if (parser.resolveOperands(valueOperands, odsBuildableType1, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetAttributeTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetDefiningOpOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetDefiningOpOpGenericAdaptorBase::GetDefiningOpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_defining_op", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetDefiningOpOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetDefiningOpOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(GetDefiningOpOp op) : GetDefiningOpOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::PDLType> GetDefiningOpOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetDefiningOpOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSResults(0).begin());
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetDefiningOpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetDefiningOpOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetOperandOpGenericAdaptorBase::GetOperandOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_operand", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetOperandOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetOperandOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandOpGenericAdaptorBase::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetOperandOp::getIndexAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetOperandOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetOperandOpAdaptor::GetOperandOpAdaptor(GetOperandOp op) : GetOperandOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetOperandOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.get_operand' op ""requires attribute 'index'");
    if (namedAttrIt->getName() == GetOperandOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_operand' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetOperandOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::ValueType> GetOperandOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::ValueType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr GetOperandOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t GetOperandOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetOperandOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetOperandOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(*this, v.getType(), "result", index++)))
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetOperandOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetOperandsOpGenericAdaptorBase::GetOperandsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_operands", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetOperandsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetOperandsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandsOpGenericAdaptorBase::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetOperandsOp::getIndexAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint32_t> GetOperandsOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetOperandsOpAdaptor::GetOperandsOpAdaptor(GetOperandsOp op) : GetOperandsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetOperandsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GetOperandsOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_operands' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetOperandsOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::PDLType> GetOperandsOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr GetOperandsOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint32_t> GetOperandsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetOperandsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetOperandsOp::setIndex(::std::optional<uint32_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue));
    (*this)->removeAttr(getIndexAttrName());
}

::mlir::Attribute GetOperandsOp::removeIndexAttr() {
  return (*this)->removeAttr(getIndexAttrName());
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, std::optional<unsigned> index) {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
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
    if (parseResult.has_value() && failed(*parseResult))
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetOperandsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultOpGenericAdaptorBase::GetResultOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_result", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetResultOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetResultOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultOpGenericAdaptorBase::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetResultOp::getIndexAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetResultOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp op) : GetResultOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.get_result' op ""requires attribute 'index'");
    if (namedAttrIt->getName() == GetResultOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetResultOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::ValueType> GetResultOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::ValueType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr GetResultOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t GetResultOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetResultOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(*this, v.getType(), "result", index++)))
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetResultOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultsOpGenericAdaptorBase::GetResultsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_results", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetResultsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetResultsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultsOpGenericAdaptorBase::getIndexAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetResultsOp::getIndexAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint32_t> GetResultsOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetResultsOpAdaptor::GetResultsOpAdaptor(GetResultsOp op) : GetResultsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetResultsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_index;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GetResultsOp::getIndexAttrName(*odsOpName)) {
      tblgen_index = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_index && !((((tblgen_index.isa<::mlir::IntegerAttr>())) && ((tblgen_index.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_index.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetResultsOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::PDLType> GetResultsOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr GetResultsOp::getIndexAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIndexAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint32_t> GetResultsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetResultsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetResultsOp::setIndex(::std::optional<uint32_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue));
    (*this)->removeAttr(getIndexAttrName());
}

::mlir::Attribute GetResultsOp::removeIndexAttr() {
  return (*this)->removeAttr(getIndexAttrName());
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, std::optional<unsigned> index) {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
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
    if (parseResult.has_value() && failed(*parseResult))
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetResultsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetUsersOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetUsersOpGenericAdaptorBase::GetUsersOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_users", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetUsersOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetUsersOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetUsersOpAdaptor::GetUsersOpAdaptor(GetUsersOp op) : GetUsersOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::PDLType> GetUsersOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::RangeType> GetUsersOp::getOperations() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps10(*this, v.getType(), "result", index++)))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetUsersOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetUsersOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetValueTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetValueTypeOpGenericAdaptorBase::GetValueTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_value_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetValueTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetValueTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(GetValueTypeOp op) : GetValueTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::PDLType> GetValueTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::pdl::PDLType> GetValueTypeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetValueTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetValueTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::IsNotNullOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
IsNotNullOpGenericAdaptorBase::IsNotNullOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.is_not_null", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> IsNotNullOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr IsNotNullOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
IsNotNullOpAdaptor::IsNotNullOpAdaptor(IsNotNullOp op) : IsNotNullOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::PDLType> IsNotNullOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
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
    if (firstSucc.has_value()) {
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void IsNotNullOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::IsNotNullOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::RecordMatchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RecordMatchOpGenericAdaptorBase::RecordMatchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.record_match", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RecordMatchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, RecordMatchOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr RecordMatchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr RecordMatchOpGenericAdaptorBase::getRewriterAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, RecordMatchOp::getRewriterAttrName(*odsOpName)).cast<::mlir::SymbolRefAttr>();
  return attr;
}

::mlir::SymbolRefAttr RecordMatchOpGenericAdaptorBase::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOpGenericAdaptorBase::getRootKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, RecordMatchOp::getRootKindAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > RecordMatchOpGenericAdaptorBase::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr RecordMatchOpGenericAdaptorBase::getGeneratedOpsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, RecordMatchOp::getGeneratedOpsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > RecordMatchOpGenericAdaptorBase::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::IntegerAttr RecordMatchOpGenericAdaptorBase::getBenefitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RecordMatchOp::getBenefitAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint16_t RecordMatchOpGenericAdaptorBase::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
RecordMatchOpAdaptor::RecordMatchOpAdaptor(RecordMatchOp op) : RecordMatchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult RecordMatchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_benefit;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'benefit'");
    if (namedAttrIt->getName() == RecordMatchOp::getBenefitAttrName(*odsOpName)) {
      tblgen_benefit = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_generatedOps;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RecordMatchOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RecordMatchOp::getGeneratedOpsAttrName(*odsOpName)) {
      tblgen_generatedOps = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rewriter;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'rewriter'");
    if (namedAttrIt->getName() == RecordMatchOp::getRewriterAttrName(*odsOpName)) {
      tblgen_rewriter = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rootKind;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == RecordMatchOp::getRootKindAttrName(*odsOpName)) {
      tblgen_rootKind = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'pdl_interp.record_match' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_rewriter && !((tblgen_rewriter.isa<::mlir::SymbolRefAttr>())))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rewriter' failed to satisfy constraint: symbol reference attribute");

  if (tblgen_rootKind && !((tblgen_rootKind.isa<::mlir::StringAttr>())))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");

  if (tblgen_generatedOps && !(((tblgen_generatedOps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_generatedOps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'generatedOps' failed to satisfy constraint: string array attribute");

  if (tblgen_benefit && !((((tblgen_benefit.isa<::mlir::IntegerAttr>())) && ((tblgen_benefit.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(16)))) && ((!tblgen_benefit.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
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
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RecordMatchOp::getMatchedOpsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getRewriterAttrName()).cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr RecordMatchOp::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOp::getRootKindAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getRootKindAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > RecordMatchOp::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr RecordMatchOp::getGeneratedOpsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getGeneratedOpsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > RecordMatchOp::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::IntegerAttr RecordMatchOp::getBenefitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBenefitAttrName()).cast<::mlir::IntegerAttr>();
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

void RecordMatchOp::setRootKind(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getRootKindAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getRootKindAttrName());
}

void RecordMatchOp::setGeneratedOpsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getGeneratedOpsAttrName(), attr);
}

void RecordMatchOp::setBenefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getBenefitAttrName(), attr);
}

void RecordMatchOp::setBenefit(uint16_t attrValue) {
  (*this)->setAttr(getBenefitAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(16), attrValue));
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
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
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
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
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
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
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
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
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
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_generatedOps;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getGeneratedOpsAttrName()) {
      tblgen_generatedOps = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rewriter;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'rewriter'");
    if (namedAttrIt->getName() == getRewriterAttrName()) {
      tblgen_rewriter = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rootKind;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getRootKindAttrName()) {
      tblgen_rootKind = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps9(*this, tblgen_rewriter, "rewriter")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_rootKind, "rootKind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_generatedOps, "generatedOps")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(*this, tblgen_benefit, "benefit")))
    return ::mlir::failure();
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::ArrayAttr generatedOpsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> matchedOpsOperands;
  ::llvm::SMLoc matchedOpsOperandsLoc;
  (void)matchedOpsOperandsLoc;
  ::mlir::StringAttr rootKindAttr;
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseCustomAttributeWithFallback(rewriterAttr, parser.getBuilder().getType<::mlir::NoneType>(), "rewriter",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

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
  if (::mlir::succeeded(parser.parseOptionalKeyword("generatedOps"))) {
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
  if (::mlir::succeeded(parser.parseOptionalComma())) {
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
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(inputsOperands.size()), static_cast<int32_t>(matchedOpsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matchedOpsOperands, odsBuildableType0, matchedOpsOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("rewriter");
  elidedAttrs.push_back("benefit");
  elidedAttrs.push_back("generatedOps");
  elidedAttrs.push_back("rootKind");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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

namespace detail {
ReplaceOpGenericAdaptorBase::ReplaceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.replace", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReplaceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ReplaceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp op) : ReplaceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::pdl::OperationType> ReplaceOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> replValuesTypes;

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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ReplaceOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchAttributeOpGenericAdaptorBase::SwitchAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchAttributeOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SwitchAttributeOp::getCaseValuesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchAttributeOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(SwitchAttributeOp op) : SwitchAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.switch_attribute' op ""requires attribute 'caseValues'");
    if (namedAttrIt->getName() == SwitchAttributeOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_caseValues && !((tblgen_caseValues.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'pdl_interp.switch_attribute' op ""attribute 'caseValues' failed to satisfy constraint: array attribute");
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

::mlir::TypedValue<::mlir::pdl::AttributeType> SwitchAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'caseValues'");
    if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, attributeOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperandCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOperandCountOpGenericAdaptorBase::SwitchOperandCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_operand_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchOperandCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchOperandCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SwitchOperandCountOp::getCaseValuesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(SwitchOperandCountOp op) : SwitchOperandCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchOperandCountOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.switch_operand_count' op ""requires attribute 'caseValues'");
    if (namedAttrIt->getName() == SwitchOperandCountOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_caseValues.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl_interp.switch_operand_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> SwitchOperandCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCaseValuesAttrName()).cast<::mlir::DenseIntElementsAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'caseValues'");
    if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperandCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperationNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOperationNameOpGenericAdaptorBase::SwitchOperationNameOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_operation_name", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchOperationNameOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchOperationNameOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchOperationNameOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SwitchOperationNameOp::getCaseValuesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchOperationNameOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(SwitchOperationNameOp op) : SwitchOperationNameOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchOperationNameOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.switch_operation_name' op ""requires attribute 'caseValues'");
    if (namedAttrIt->getName() == SwitchOperationNameOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'pdl_interp.switch_operation_name' op ""attribute 'caseValues' failed to satisfy constraint: string array attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> SwitchOperationNameOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'caseValues'");
    if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperationNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchResultCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchResultCountOpGenericAdaptorBase::SwitchResultCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_result_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchResultCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchResultCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SwitchResultCountOp::getCaseValuesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(SwitchResultCountOp op) : SwitchResultCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchResultCountOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.switch_result_count' op ""requires attribute 'caseValues'");
    if (namedAttrIt->getName() == SwitchResultCountOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_caseValues.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl_interp.switch_result_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
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

::mlir::TypedValue<::mlir::pdl::OperationType> SwitchResultCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCaseValuesAttrName()).cast<::mlir::DenseIntElementsAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'caseValues'");
    if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchResultCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchTypeOpGenericAdaptorBase::SwitchTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypeOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SwitchTypeOp::getCaseValuesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchTypeOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(SwitchTypeOp op) : SwitchTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchTypeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.switch_type' op ""requires attribute 'caseValues'");
    if (namedAttrIt->getName() == SwitchTypeOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))); }))))
    return emitError(loc, "'pdl_interp.switch_type' op ""attribute 'caseValues' failed to satisfy constraint: type array attribute");
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

::mlir::TypedValue<::mlir::pdl::TypeType> SwitchTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'caseValues'");
    if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchTypesOpGenericAdaptorBase::SwitchTypesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_types", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchTypesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchTypesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypesOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SwitchTypesOp::getCaseValuesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchTypesOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(SwitchTypesOp op) : SwitchTypesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchTypesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'pdl_interp.switch_types' op ""requires attribute 'caseValues'");
    if (namedAttrIt->getName() == SwitchTypesOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_caseValues && !(((tblgen_caseValues.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_caseValues.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())) && ((true))); }))); }))))
    return emitError(loc, "'pdl_interp.switch_types' op ""attribute 'caseValues' failed to satisfy constraint: type-array array attribute");
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

::mlir::TypedValue<::mlir::pdl::RangeType> SwitchTypesOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
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
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCaseValuesAttrName()).cast<::mlir::ArrayAttr>();
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
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_caseValues;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'caseValues'");
    if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_caseValues = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps13(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
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
    if (firstSucc.has_value()) {
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
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypesOp)


#endif  // GET_OP_CLASSES

