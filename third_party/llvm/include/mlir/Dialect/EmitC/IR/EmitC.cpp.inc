/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::emitc::ApplyOp,
::mlir::emitc::CallOp,
::mlir::emitc::CastOp,
::mlir::emitc::ConstantOp,
::mlir::emitc::IncludeOp,
::mlir::emitc::VariableOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace emitc {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_EmitC0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_EmitC0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_EmitC1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_EmitC2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::emitc::OpaqueAttr>())) || ((attr.isa<::mlir::TypedAttr>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: An opaque attribute or TypedAttr instance";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_EmitC3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}
} // namespace emitc
} // namespace mlir
namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::ApplyOp definitions
//===----------------------------------------------------------------------===//

ApplyOpAdaptor::ApplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("emitc.apply", odsAttrs.getContext());
}

ApplyOpAdaptor::ApplyOpAdaptor(ApplyOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ApplyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ApplyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ApplyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ApplyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyOpAdaptor::getApplicableOperatorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ApplyOp::getApplicableOperatorAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyOpAdaptor::getApplicableOperator() {
  auto attr = getApplicableOperatorAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_applicableOperator;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'emitc.apply' op ""requires attribute 'applicableOperator'");
    if (namedAttrIt->getName() == ApplyOp::getApplicableOperatorAttrName(*odsOpName)) {
      tblgen_applicableOperator = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_applicableOperator && !((tblgen_applicableOperator.isa<::mlir::StringAttr>())))
    return emitError(loc, "'emitc.apply' op ""attribute 'applicableOperator' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ApplyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ApplyOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ApplyOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr ApplyOp::getApplicableOperatorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getApplicableOperatorAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyOp::getApplicableOperator() {
  auto attr = getApplicableOperatorAttr();
  return attr.getValue();
}

void ApplyOp::setApplicableOperatorAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getApplicableOperatorAttrName(), attr);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getApplicableOperatorAttrName(odsState.name), applicableOperator);
  odsState.addTypes(result);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getApplicableOperatorAttrName(odsState.name), applicableOperator);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getApplicableOperatorAttrName(odsState.name), odsBuilder.getStringAttr(applicableOperator));
  odsState.addTypes(result);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(getApplicableOperatorAttrName(odsState.name), odsBuilder.getStringAttr(applicableOperator));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_applicableOperator;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'applicableOperator'");
    if (namedAttrIt->getName() == getApplicableOperatorAttrName()) {
      tblgen_applicableOperator = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC0(*this, tblgen_applicableOperator, "applicableOperator")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ApplyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr applicableOperatorAttr;
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> operandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(applicableOperatorAttr, parser.getBuilder().getType<::mlir::NoneType>(), "applicableOperator",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType operand__allResult_functionType;
  if (parser.parseType(operand__allResult_functionType))
    return ::mlir::failure();
  operandTypes = operand__allResult_functionType.getInputs();
  allResultTypes = operand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandOperands, operandTypes, operandOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getApplicableOperatorAttr());
  _odsPrinter << "(";
  _odsPrinter << getOperand();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"applicableOperator"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getOperand().getType()), getOperation()->getResultTypes());
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::ApplyOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::CallOp definitions
//===----------------------------------------------------------------------===//

CallOpAdaptor::CallOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("emitc.call", odsAttrs.getContext());
}

CallOpAdaptor::CallOpAdaptor(CallOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CallOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CallOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CallOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CallOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr CallOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CallOpAdaptor::getCalleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CallOp::getCalleeAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CallOpAdaptor::getCallee() {
  auto attr = getCalleeAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CallOpAdaptor::getArgsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CallOp::getArgsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CallOpAdaptor::getArgs() {
  auto attr = getArgsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CallOpAdaptor::getTemplateArgsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, CallOp::getTemplateArgsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CallOpAdaptor::getTemplateArgs() {
  auto attr = getTemplateArgsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult CallOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_callee;
  ::mlir::Attribute tblgen_args;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'emitc.call' op ""requires attribute 'callee'");
    if (namedAttrIt->getName() == CallOp::getCalleeAttrName(*odsOpName)) {
      tblgen_callee = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CallOp::getArgsAttrName(*odsOpName)) {
      tblgen_args = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_template_args;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CallOp::getTemplateArgsAttrName(*odsOpName)) {
      tblgen_template_args = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_callee && !((tblgen_callee.isa<::mlir::StringAttr>())))
    return emitError(loc, "'emitc.call' op ""attribute 'callee' failed to satisfy constraint: string attribute");

  if (tblgen_args && !((tblgen_args.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'emitc.call' op ""attribute 'args' failed to satisfy constraint: array attribute");

  if (tblgen_template_args && !((tblgen_template_args.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'emitc.call' op ""attribute 'template_args' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CallOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CallOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CallOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CallOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr CallOp::getCalleeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCalleeAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CallOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CallOp::getArgsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getArgsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CallOp::getArgs() {
  auto attr = getArgsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CallOp::getTemplateArgsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTemplateArgsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CallOp::getTemplateArgs() {
  auto attr = getTemplateArgsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void CallOp::setCalleeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

void CallOp::setArgsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgsAttrName(), attr);
}

void CallOp::setTemplateArgsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTemplateArgsAttrName(), attr);
}

::mlir::Attribute CallOp::removeArgsAttr() {
  return (*this)->removeAttr(getArgsAttrName());
}

::mlir::Attribute CallOp::removeTemplate_argsAttr() {
  return (*this)->removeAttr(getTemplateArgsAttrName());
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::StringAttr callee, /*optional*/::mlir::ArrayAttr args, /*optional*/::mlir::ArrayAttr template_args, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  if (args) {
  odsState.addAttribute(getArgsAttrName(odsState.name), args);
  }
  if (template_args) {
  odsState.addAttribute(getTemplateArgsAttrName(odsState.name), template_args);
  }
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::llvm::StringRef callee, /*optional*/::mlir::ArrayAttr args, /*optional*/::mlir::ArrayAttr template_args, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(getCalleeAttrName(odsState.name), odsBuilder.getStringAttr(callee));
  if (args) {
  odsState.addAttribute(getArgsAttrName(odsState.name), args);
  }
  if (template_args) {
  odsState.addAttribute(getTemplateArgsAttrName(odsState.name), template_args);
  }
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CallOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_callee;
  ::mlir::Attribute tblgen_args;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'callee'");
    if (namedAttrIt->getName() == getCalleeAttrName()) {
      tblgen_callee = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getArgsAttrName()) {
      tblgen_args = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_template_args;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getTemplateArgsAttrName()) {
      tblgen_template_args = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC0(*this, tblgen_callee, "callee")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC1(*this, tblgen_args, "args")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC1(*this, tblgen_template_args, "template_args")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CallOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr calleeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> operandsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "callee",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseLParen())
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType operands__allResult_functionType;
  if (parser.parseType(operands__allResult_functionType))
    return ::mlir::failure();
  operandsTypes = operands__allResult_functionType.getInputs();
  allResultTypes = operands__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CallOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCalleeAttr());
  _odsPrinter << "(";
  _odsPrinter << operands();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"callee"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(operands().getTypes(), getOperation()->getResultTypes());
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::CallOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::CastOp definitions
//===----------------------------------------------------------------------===//

CastOpAdaptor::CastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("emitc.cast", odsAttrs.getContext());
}

CastOpAdaptor::CastOpAdaptor(CastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CastOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CastOpAdaptor::getAttributes() {
  return odsAttrs;
}

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

::mlir::Value CastOp::getSource() {
  return *getODSOperands(0).begin();
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

::mlir::Value CastOp::getDest() {
  return *getODSResults(0).begin();
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "result", index++)))
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
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::CastOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("emitc.constant", odsAttrs.getContext());
}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConstantOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstantOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ConstantOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute ConstantOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantOp::getValueAttrName(*odsOpName)).cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute ConstantOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'emitc.constant' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !(((tblgen_value.isa<::mlir::emitc::OpaqueAttr>())) || ((tblgen_value.isa<::mlir::TypedAttr>()))))
    return emitError(loc, "'emitc.constant' op ""attribute 'value' failed to satisfy constraint: An opaque attribute or TypedAttr instance");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Attribute ConstantOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC2(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::ConstantOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::IncludeOp definitions
//===----------------------------------------------------------------------===//

IncludeOpAdaptor::IncludeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("emitc.include", odsAttrs.getContext());
}

IncludeOpAdaptor::IncludeOpAdaptor(IncludeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IncludeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IncludeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IncludeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr IncludeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr IncludeOpAdaptor::getIncludeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, IncludeOp::getIncludeAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef IncludeOpAdaptor::getInclude() {
  auto attr = getIncludeAttr();
  return attr.getValue();
}

::mlir::UnitAttr IncludeOpAdaptor::getIsStandardIncludeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, IncludeOp::getIsStandardIncludeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool IncludeOpAdaptor::getIsStandardInclude() {
  auto attr = getIsStandardIncludeAttr();
  return attr != nullptr;
}

::mlir::LogicalResult IncludeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_include;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'emitc.include' op ""requires attribute 'include'");
    if (namedAttrIt->getName() == IncludeOp::getIncludeAttrName(*odsOpName)) {
      tblgen_include = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_is_standard_include;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == IncludeOp::getIsStandardIncludeAttrName(*odsOpName)) {
      tblgen_is_standard_include = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_include && !((tblgen_include.isa<::mlir::StringAttr>())))
    return emitError(loc, "'emitc.include' op ""attribute 'include' failed to satisfy constraint: string attribute");

  if (tblgen_is_standard_include && !((tblgen_is_standard_include.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'emitc.include' op ""attribute 'is_standard_include' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IncludeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IncludeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> IncludeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IncludeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr IncludeOp::getIncludeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIncludeAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef IncludeOp::getInclude() {
  auto attr = getIncludeAttr();
  return attr.getValue();
}

::mlir::UnitAttr IncludeOp::getIsStandardIncludeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getIsStandardIncludeAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool IncludeOp::getIsStandardInclude() {
  auto attr = getIsStandardIncludeAttr();
  return attr != nullptr;
}

void IncludeOp::setIncludeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getIncludeAttrName(), attr);
}

void IncludeOp::setIsStandardIncludeAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIsStandardIncludeAttrName(), attr);
}

::mlir::Attribute IncludeOp::removeIs_standard_includeAttr() {
  return (*this)->removeAttr(getIsStandardIncludeAttrName());
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr include, /*optional*/::mlir::UnitAttr is_standard_include) {
  odsState.addAttribute(getIncludeAttrName(odsState.name), include);
  if (is_standard_include) {
  odsState.addAttribute(getIsStandardIncludeAttrName(odsState.name), is_standard_include);
  }
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr include, /*optional*/::mlir::UnitAttr is_standard_include) {
  odsState.addAttribute(getIncludeAttrName(odsState.name), include);
  if (is_standard_include) {
  odsState.addAttribute(getIsStandardIncludeAttrName(odsState.name), is_standard_include);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef include, /*optional*/bool is_standard_include) {
  odsState.addAttribute(getIncludeAttrName(odsState.name), odsBuilder.getStringAttr(include));
  if (is_standard_include) {
  odsState.addAttribute(getIsStandardIncludeAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef include, /*optional*/bool is_standard_include) {
  odsState.addAttribute(getIncludeAttrName(odsState.name), odsBuilder.getStringAttr(include));
  if (is_standard_include) {
  odsState.addAttribute(getIsStandardIncludeAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IncludeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IncludeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_include;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'include'");
    if (namedAttrIt->getName() == getIncludeAttrName()) {
      tblgen_include = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_is_standard_include;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getIsStandardIncludeAttrName()) {
      tblgen_is_standard_include = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC0(*this, tblgen_include, "include")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC3(*this, tblgen_is_standard_include, "is_standard_include")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult IncludeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void IncludeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::IncludeOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::VariableOp definitions
//===----------------------------------------------------------------------===//

VariableOpAdaptor::VariableOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("emitc.variable", odsAttrs.getContext());
}

VariableOpAdaptor::VariableOpAdaptor(VariableOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VariableOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VariableOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VariableOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr VariableOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute VariableOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, VariableOp::getValueAttrName(*odsOpName)).cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute VariableOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::LogicalResult VariableOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'emitc.variable' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == VariableOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !(((tblgen_value.isa<::mlir::emitc::OpaqueAttr>())) || ((tblgen_value.isa<::mlir::TypedAttr>()))))
    return emitError(loc, "'emitc.variable' op ""attribute 'value' failed to satisfy constraint: An opaque attribute or TypedAttr instance");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VariableOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VariableOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> VariableOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VariableOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Attribute VariableOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute VariableOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void VariableOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void VariableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType0);
}

void VariableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VariableOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VariableOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC2(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VariableOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace emitc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::VariableOp)


#endif  // GET_OP_CLASSES

