/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::cf::AssertOp,
::mlir::cf::BranchOp,
::mlir::cf::CondBranchOp,
::mlir::cf::SwitchOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace cf {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ControlFlowOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ControlFlowOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ControlFlowOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((true)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ControlFlowOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseI32ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i32 dense array attribute";
  }
  return ::mlir::success();
}
} // namespace cf
} // namespace mlir
namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::AssertOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AssertOpGenericAdaptorBase::AssertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("cf.assert", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AssertOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AssertOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AssertOpGenericAdaptorBase::getMsgAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AssertOp::getMsgAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef AssertOpGenericAdaptorBase::getMsg() {
  auto attr = getMsgAttr();
  return attr.getValue();
}

} // namespace detail
AssertOpAdaptor::AssertOpAdaptor(AssertOp op) : AssertOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AssertOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_msg;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'cf.assert' op ""requires attribute 'msg'");
    if (namedAttrIt->getName() == AssertOp::getMsgAttrName(*odsOpName)) {
      tblgen_msg = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_msg && !((tblgen_msg.isa<::mlir::StringAttr>())))
    return emitError(loc, "'cf.assert' op ""attribute 'msg' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AssertOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AssertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> AssertOp::getArg() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange AssertOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AssertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AssertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AssertOp::getMsgAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMsgAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AssertOp::getMsg() {
  auto attr = getMsgAttr();
  return attr.getValue();
}

void AssertOp::setMsgAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getMsgAttrName(), attr);
}

void AssertOp::setMsg(::llvm::StringRef attrValue) {
  (*this)->setAttr(getMsgAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr msg) {
  odsState.addOperands(arg);
  odsState.addAttribute(getMsgAttrName(odsState.name), msg);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr msg) {
  odsState.addOperands(arg);
  odsState.addAttribute(getMsgAttrName(odsState.name), msg);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::llvm::StringRef msg) {
  odsState.addOperands(arg);
  odsState.addAttribute(getMsgAttrName(odsState.name), odsBuilder.getStringAttr(msg));
}

void AssertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef msg) {
  odsState.addOperands(arg);
  odsState.addAttribute(getMsgAttrName(odsState.name), odsBuilder.getStringAttr(msg));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AssertOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_msg;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'msg'");
    if (namedAttrIt->getName() == getMsgAttrName()) {
      tblgen_msg = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps0(*this, tblgen_msg, "msg")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AssertOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void AssertOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::ParseResult AssertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::StringAttr msgAttr;

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(msgAttr, parser.getBuilder().getType<::mlir::NoneType>(), "msg",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(argOperands, odsBuildableType0, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AssertOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMsgAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("msg");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::AssertOp)

namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::BranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BranchOpGenericAdaptorBase::BranchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("cf.br", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr BranchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BranchOpAdaptor::BranchOpAdaptor(BranchOp op) : BranchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BranchOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range BranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BranchOp::getDestOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange BranchOp::getDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
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

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Block *dest, ValueRange destOperands) {
      odsState.addSuccessors(dest);
      odsState.addOperands(destOperands);
    
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BranchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult BranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void BranchOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> destOperandsOperands;
  ::llvm::SMLoc destOperandsOperandsLoc;
  (void)destOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> destOperandsTypes;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  destOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(destOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(destOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  if (!getDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getDestOperands().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void BranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::BranchOp)

namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::CondBranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CondBranchOpGenericAdaptorBase::CondBranchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("cf.cond_br", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CondBranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CondBranchOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CondBranchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CondBranchOpAdaptor::CondBranchOpAdaptor(CondBranchOp op) : CondBranchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CondBranchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'cf.cond_br' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CondBranchOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'cf.cond_br' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> CondBranchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CondBranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> CondBranchOp::getCondition() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range CondBranchOp::getTrueDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CondBranchOp::getFalseDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CondBranchOp::getConditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CondBranchOp::getTrueDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CondBranchOp::getFalseDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CondBranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CondBranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CondBranchOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CondBranchOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, ValueRange trueOperands, Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CondBranchOp::verifyInvariantsImpl() {
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CondBranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CondBranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand conditionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> conditionOperands(conditionRawOperands);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::Block *trueDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> trueDestOperandsOperands;
  ::llvm::SMLoc trueDestOperandsOperandsLoc;
  (void)trueDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> trueDestOperandsTypes;
  ::mlir::Block *falseDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> falseDestOperandsOperands;
  ::llvm::SMLoc falseDestOperandsOperandsLoc;
  (void)falseDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> falseDestOperandsTypes;

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(trueDestSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(trueDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(falseDestSuccessor))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(falseDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, conditionOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CondBranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCondition();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTrueDest();
  if (!getTrueDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getTrueDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getTrueDestOperands().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFalseDest();
  if (!getFalseDestOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getFalseDestOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getFalseDestOperands().getTypes();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CondBranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::CondBranchOp)

namespace mlir {
namespace cf {

//===----------------------------------------------------------------------===//
// ::mlir::cf::SwitchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOpGenericAdaptorBase::SwitchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("cf.switch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SwitchOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SwitchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchOpGenericAdaptorBase::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, SwitchOp::getCaseValuesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::std::optional< ::mlir::DenseIntElementsAttr > SwitchOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::mlir::DenseI32ArrayAttr SwitchOpGenericAdaptorBase::getCaseOperandSegmentsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SwitchOp::getCaseOperandSegmentsAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int32_t> SwitchOpGenericAdaptorBase::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

} // namespace detail
SwitchOpAdaptor::SwitchOpAdaptor(SwitchOp op) : SwitchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SwitchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_case_operand_segments;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'cf.switch' op ""requires attribute 'case_operand_segments'");
    if (namedAttrIt->getName() == SwitchOp::getCaseOperandSegmentsAttrName(*odsOpName)) {
      tblgen_case_operand_segments = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_case_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'cf.switch' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SwitchOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SwitchOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_case_values = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'cf.switch' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_case_values && !(((tblgen_case_values.isa<::mlir::DenseIntElementsAttr>())) && ((true))))
    return emitError(loc, "'cf.switch' op ""attribute 'case_values' failed to satisfy constraint: integer elements attribute");

  if (tblgen_case_operand_segments && !((tblgen_case_operand_segments.isa<::mlir::DenseI32ArrayAttr>())))
    return emitError(loc, "'cf.switch' op ""attribute 'case_operand_segments' failed to satisfy constraint: i32 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SwitchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> SwitchOp::getFlag() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range SwitchOp::getDefaultOperands() {
  return getODSOperands(1);
}

::mlir::OperandRangeRange SwitchOp::getCaseOperands() {
  return getODSOperands(2).split(getCaseOperandSegmentsAttr());
}

::mlir::MutableOperandRange SwitchOp::getFlagMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SwitchOp::getDefaultOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRangeRange SwitchOp::getCaseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange.split(*(*this)->getAttrDictionary().getNamed(getCaseOperandSegmentsAttrName()));
}

std::pair<unsigned, unsigned> SwitchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOp::getDefaultDestination() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOp::getCaseDestinations() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::DenseIntElementsAttr SwitchOp::getCaseValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getCaseValuesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::std::optional< ::mlir::DenseIntElementsAttr > SwitchOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::std::optional< ::mlir::DenseIntElementsAttr >(attr) : (::std::nullopt);
}

::mlir::DenseI32ArrayAttr SwitchOp::getCaseOperandSegmentsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCaseOperandSegmentsAttrName()).cast<::mlir::DenseI32ArrayAttr>();
}

::llvm::ArrayRef<int32_t> SwitchOp::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

void SwitchOp::setCaseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOp::setCaseOperandSegmentsAttr(::mlir::DenseI32ArrayAttr attr) {
  (*this)->setAttr(getCaseOperandSegmentsAttrName(), attr);
}

void SwitchOp::setCaseOperandSegments(::llvm::ArrayRef<int32_t> attrValue) {
  (*this)->setAttr(getCaseOperandSegmentsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI32ArrayAttr(attrValue));
}

::mlir::Attribute SwitchOp::removeCaseValuesAttr() {
  return (*this)->removeAttr(getCaseValuesAttrName());
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value flag, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::DenseIntElementsAttr case_values, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(flag);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    odsState.addAttribute(getCaseOperandSegmentsAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr(rangeSegments));  }
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + range.size(); }))}));
  if (case_values) {
    odsState.addAttribute(getCaseValuesAttrName(odsState.name), case_values);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value flag, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::DenseIntElementsAttr case_values, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(flag);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    odsState.addAttribute(getCaseOperandSegmentsAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr(rangeSegments));  }
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + range.size(); }))}));
  if (case_values) {
    odsState.addAttribute(getCaseValuesAttrName(odsState.name), case_values);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_case_operand_segments;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'case_operand_segments'");
    if (namedAttrIt->getName() == getCaseOperandSegmentsAttrName()) {
      tblgen_case_operand_segments = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_case_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_case_values = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps1(*this, tblgen_case_values, "case_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ControlFlowOps2(*this, tblgen_case_operand_segments, "case_operand_segments")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);
    if (::mlir::failed(::mlir::OpTrait::impl::verifyValueSizeAttr(*this, "case_operand_segments", "caseOperands", valueGroup2.size())))
      return ::mlir::failure();

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ControlFlowOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand flagRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> flagOperands(flagRawOperands);  ::llvm::SMLoc flagOperandsLoc;
  (void)flagOperandsLoc;
  ::mlir::Type flagRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> flagTypes(flagRawTypes);
  ::mlir::Block *defaultDestinationSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> defaultOperandsOperands;
  ::llvm::SMLoc defaultOperandsOperandsLoc;
  (void)defaultOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> defaultOperandsTypes;
  ::mlir::DenseIntElementsAttr case_valuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> caseDestinationsSuccessors;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> caseOperandsOperands;
    llvm::SmallVector<int32_t> caseOperandsOperandGroupSizes;
  ::llvm::SMLoc caseOperandsOperandsLoc;
  (void)caseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> caseOperandsTypes;

  flagOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(flagRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    flagRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    defaultOperandsOperandsLoc = parser.getCurrentLocation();
    caseOperandsOperandsLoc = parser.getCurrentLocation();
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand>> caseOperandsOperandGroups;
    ::llvm::SmallVector<llvm::SmallVector<::mlir::Type>> caseOperandsTypeGroups;
    if (parseSwitchOpCases(parser, flagRawTypes[0], defaultDestinationSuccessor, defaultOperandsOperands, defaultOperandsTypes, case_valuesAttr, caseDestinationsSuccessors, caseOperandsOperandGroups, caseOperandsTypeGroups))
      return ::mlir::failure();
    if (case_valuesAttr)
      result.addAttribute("case_values", case_valuesAttr);
    for (const auto &subRange : caseOperandsOperandGroups) {
      caseOperandsOperands.append(subRange.begin(), subRange.end());
      caseOperandsOperandGroupSizes.push_back(subRange.size());
    }
    for (const auto &subRange : caseOperandsTypeGroups)
      caseOperandsTypes.append(subRange.begin(), subRange.end());
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(defaultDestinationSuccessor);
  result.addSuccessors(caseDestinationsSuccessors);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(defaultOperandsOperands.size()), static_cast<int32_t>(caseOperandsOperands.size())}));
  result.addAttribute("case_operand_segments", parser.getBuilder().getDenseI32ArrayAttr(caseOperandsOperandGroupSizes));
  if (parser.resolveOperands(flagOperands, flagTypes, flagOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(defaultOperandsOperands, defaultOperandsTypes, defaultOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(caseOperandsOperands, caseOperandsTypes, caseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getFlag();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getFlag().getType();
    if (auto validType = type.dyn_cast<::mlir::IntegerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ' << "[";
  _odsPrinter.printNewline();
  printSwitchOpCases(_odsPrinter, *this, getFlag().getType(), getDefaultDestination(), getDefaultOperands(), getDefaultOperands().getTypes(), getCaseValuesAttr(), getCaseDestinations(), getCaseOperands(), getCaseOperands().getTypes());
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("case_operand_segments");
  elidedAttrs.push_back("case_values");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void SwitchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace cf
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::cf::SwitchOp)


#endif  // GET_OP_CLASSES

