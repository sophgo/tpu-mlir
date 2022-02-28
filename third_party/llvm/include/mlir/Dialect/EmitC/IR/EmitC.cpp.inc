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
::mlir::emitc::ConstantOp,
::mlir::emitc::IncludeOp
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
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
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

ApplyOpAdaptor::ApplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ApplyOpAdaptor::ApplyOpAdaptor(ApplyOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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

::mlir::Value ApplyOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ApplyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyOpAdaptor::applicableOperatorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("applicableOperator").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyOpAdaptor::applicableOperator() {
  auto attr = applicableOperatorAttr();
  return attr.getValue();
}

::mlir::LogicalResult ApplyOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_applicableOperator = odsAttrs.get("applicableOperator");
    if (!tblgen_applicableOperator)
      return emitError(loc, "'emitc.apply' op ""requires attribute 'applicableOperator'");

    if (tblgen_applicableOperator && !((tblgen_applicableOperator.isa<::mlir::StringAttr>())))
      return emitError(loc, "'emitc.apply' op ""attribute 'applicableOperator' failed to satisfy constraint: string attribute");
  }
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

::mlir::Value ApplyOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ApplyOp::operandMutable() {
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

::mlir::Value ApplyOp::result() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr ApplyOp::applicableOperatorAttr() {
  return (*this)->getAttr(applicableOperatorAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyOp::applicableOperator() {
  auto attr = applicableOperatorAttr();
  return attr.getValue();
}

void ApplyOp::applicableOperatorAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(applicableOperatorAttrName(), attr);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::StringAttr applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(applicableOperatorAttrName(odsState.name), applicableOperator);
  odsState.addTypes(result);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(applicableOperatorAttrName(odsState.name), applicableOperator);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(applicableOperatorAttrName(odsState.name), odsBuilder.getStringAttr(applicableOperator));
  odsState.addTypes(result);
}

void ApplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef applicableOperator, ::mlir::Value operand) {
  odsState.addOperands(operand);
  odsState.addAttribute(applicableOperatorAttrName(odsState.name), odsBuilder.getStringAttr(applicableOperator));
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

::mlir::LogicalResult ApplyOp::verifyInvariants() {
  {
    auto tblgen_applicableOperator = (*this)->getAttr(applicableOperatorAttrName());
    if (!tblgen_applicableOperator)
      return emitOpError("requires attribute 'applicableOperator'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC0(*this, tblgen_applicableOperator, "applicableOperator")))
      return ::mlir::failure();
  }
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
  return verify();
}

::mlir::ParseResult ApplyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr applicableOperatorAttr;
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
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
  _odsPrinter.printAttributeWithoutType(applicableOperatorAttr());
  _odsPrinter << "(";
  _odsPrinter << operand();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"applicableOperator"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(operand().getType()), getOperation()->getResultTypes());
}

} // namespace emitc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::ApplyOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::CallOp definitions
//===----------------------------------------------------------------------===//

CallOpAdaptor::CallOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CallOpAdaptor::CallOpAdaptor(CallOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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

::mlir::StringAttr CallOpAdaptor::calleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("callee").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CallOpAdaptor::callee() {
  auto attr = calleeAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CallOpAdaptor::argsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("args").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CallOpAdaptor::args() {
  auto attr = argsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CallOpAdaptor::template_argsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("template_args").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CallOpAdaptor::template_args() {
  auto attr = template_argsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult CallOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_callee = odsAttrs.get("callee");
    if (!tblgen_callee)
      return emitError(loc, "'emitc.call' op ""requires attribute 'callee'");

    if (tblgen_callee && !((tblgen_callee.isa<::mlir::StringAttr>())))
      return emitError(loc, "'emitc.call' op ""attribute 'callee' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_args = odsAttrs.get("args");
    if (tblgen_args && !((tblgen_args.isa<::mlir::ArrayAttr>())))
      return emitError(loc, "'emitc.call' op ""attribute 'args' failed to satisfy constraint: array attribute");
  }
  {
    auto tblgen_template_args = odsAttrs.get("template_args");
    if (tblgen_template_args && !((tblgen_template_args.isa<::mlir::ArrayAttr>())))
      return emitError(loc, "'emitc.call' op ""attribute 'template_args' failed to satisfy constraint: array attribute");
  }
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

::mlir::StringAttr CallOp::calleeAttr() {
  return (*this)->getAttr(calleeAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CallOp::callee() {
  auto attr = calleeAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CallOp::argsAttr() {
  return (*this)->getAttr(argsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CallOp::args() {
  auto attr = argsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CallOp::template_argsAttr() {
  return (*this)->getAttr(template_argsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CallOp::template_args() {
  auto attr = template_argsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void CallOp::calleeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(calleeAttrName(), attr);
}

void CallOp::argsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(argsAttrName(), attr);
}

void CallOp::template_argsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(template_argsAttrName(), attr);
}

::mlir::Attribute CallOp::removeArgsAttr() {
  return (*this)->removeAttr(argsAttrName());
}

::mlir::Attribute CallOp::removeTemplate_argsAttr() {
  return (*this)->removeAttr(template_argsAttrName());
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::mlir::StringAttr callee, /*optional*/::mlir::ArrayAttr args, /*optional*/::mlir::ArrayAttr template_args, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(calleeAttrName(odsState.name), callee);
  if (args) {
  odsState.addAttribute(argsAttrName(odsState.name), args);
  }
  if (template_args) {
  odsState.addAttribute(template_argsAttrName(odsState.name), template_args);
  }
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, ::llvm::StringRef callee, /*optional*/::mlir::ArrayAttr args, /*optional*/::mlir::ArrayAttr template_args, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute(calleeAttrName(odsState.name), odsBuilder.getStringAttr(callee));
  if (args) {
  odsState.addAttribute(argsAttrName(odsState.name), args);
  }
  if (template_args) {
  odsState.addAttribute(template_argsAttrName(odsState.name), template_args);
  }
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CallOp::verifyInvariants() {
  {
    auto tblgen_callee = (*this)->getAttr(calleeAttrName());
    if (!tblgen_callee)
      return emitOpError("requires attribute 'callee'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC0(*this, tblgen_callee, "callee")))
      return ::mlir::failure();
  }
  {
    auto tblgen_args = (*this)->getAttr(argsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC1(*this, tblgen_args, "args")))
      return ::mlir::failure();
  }
  {
    auto tblgen_template_args = (*this)->getAttr(template_argsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC1(*this, tblgen_template_args, "template_args")))
      return ::mlir::failure();
  }
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
  return verify();
}

::mlir::ParseResult CallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr calleeAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
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
  _odsPrinter.printAttributeWithoutType(calleeAttr());
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::CallOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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

::mlir::Attribute ConstantOpAdaptor::valueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::Attribute attr = odsAttrs.get("value").cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute ConstantOpAdaptor::value() {
  auto attr = valueAttr();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_value = odsAttrs.get("value");
    if (!tblgen_value)
      return emitError(loc, "'emitc.constant' op ""requires attribute 'value'");

    if (tblgen_value && !((true)))
      return emitError(loc, "'emitc.constant' op ""attribute 'value' failed to satisfy constraint: any attribute");
  }
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

::mlir::Attribute ConstantOp::valueAttr() {
  return (*this)->getAttr(valueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::valueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(valueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Attribute value) {
  odsState.addAttribute(valueAttrName(odsState.name), value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute(valueAttrName(odsState.name), value);
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

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  {
    auto tblgen_value = (*this)->getAttr(valueAttrName());
    if (!tblgen_value)
      return emitOpError("requires attribute 'value'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC2(*this, tblgen_value, "value")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_EmitC0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return verify();
}

} // namespace emitc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::ConstantOp)

namespace mlir {
namespace emitc {

//===----------------------------------------------------------------------===//
// ::mlir::emitc::IncludeOp definitions
//===----------------------------------------------------------------------===//

IncludeOpAdaptor::IncludeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

IncludeOpAdaptor::IncludeOpAdaptor(IncludeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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

::mlir::StringAttr IncludeOpAdaptor::includeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::StringAttr attr = odsAttrs.get("include").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef IncludeOpAdaptor::include() {
  auto attr = includeAttr();
  return attr.getValue();
}

::mlir::UnitAttr IncludeOpAdaptor::is_standard_includeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("is_standard_include").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool IncludeOpAdaptor::is_standard_include() {
  auto attr = is_standard_includeAttr();
  return attr != nullptr;
}

::mlir::LogicalResult IncludeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_include = odsAttrs.get("include");
    if (!tblgen_include)
      return emitError(loc, "'emitc.include' op ""requires attribute 'include'");

    if (tblgen_include && !((tblgen_include.isa<::mlir::StringAttr>())))
      return emitError(loc, "'emitc.include' op ""attribute 'include' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_is_standard_include = odsAttrs.get("is_standard_include");
    if (tblgen_is_standard_include && !((tblgen_is_standard_include.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'emitc.include' op ""attribute 'is_standard_include' failed to satisfy constraint: unit attribute");
  }
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

::mlir::StringAttr IncludeOp::includeAttr() {
  return (*this)->getAttr(includeAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef IncludeOp::include() {
  auto attr = includeAttr();
  return attr.getValue();
}

::mlir::UnitAttr IncludeOp::is_standard_includeAttr() {
  return (*this)->getAttr(is_standard_includeAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool IncludeOp::is_standard_include() {
  auto attr = is_standard_includeAttr();
  return attr != nullptr;
}

void IncludeOp::includeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(includeAttrName(), attr);
}

void IncludeOp::is_standard_includeAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(is_standard_includeAttrName(), attr);
}

::mlir::Attribute IncludeOp::removeIs_standard_includeAttr() {
  return (*this)->removeAttr(is_standard_includeAttrName());
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr include, /*optional*/::mlir::UnitAttr is_standard_include) {
  odsState.addAttribute(includeAttrName(odsState.name), include);
  if (is_standard_include) {
  odsState.addAttribute(is_standard_includeAttrName(odsState.name), is_standard_include);
  }
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr include, /*optional*/::mlir::UnitAttr is_standard_include) {
  odsState.addAttribute(includeAttrName(odsState.name), include);
  if (is_standard_include) {
  odsState.addAttribute(is_standard_includeAttrName(odsState.name), is_standard_include);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef include, /*optional*/bool is_standard_include) {
  odsState.addAttribute(includeAttrName(odsState.name), odsBuilder.getStringAttr(include));
  if (is_standard_include) {
  odsState.addAttribute(is_standard_includeAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void IncludeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef include, /*optional*/bool is_standard_include) {
  odsState.addAttribute(includeAttrName(odsState.name), odsBuilder.getStringAttr(include));
  if (is_standard_include) {
  odsState.addAttribute(is_standard_includeAttrName(odsState.name), odsBuilder.getUnitAttr());
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

::mlir::LogicalResult IncludeOp::verifyInvariants() {
  {
    auto tblgen_include = (*this)->getAttr(includeAttrName());
    if (!tblgen_include)
      return emitOpError("requires attribute 'include'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC0(*this, tblgen_include, "include")))
      return ::mlir::failure();
  }
  {
    auto tblgen_is_standard_include = (*this)->getAttr(is_standard_includeAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_EmitC3(*this, tblgen_is_standard_include, "is_standard_include")))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void IncludeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace emitc
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::emitc::IncludeOp)


#endif  // GET_OP_CLASSES

