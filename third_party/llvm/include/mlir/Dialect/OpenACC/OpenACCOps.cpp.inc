/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::acc::DataOp,
::mlir::acc::EnterDataOp,
::mlir::acc::ExitDataOp,
::mlir::acc::InitOp,
::mlir::acc::LoopOp,
::mlir::acc::ParallelOp,
::mlir::acc::ShutdownOp,
::mlir::acc::TerminatorOp,
::mlir::acc::UpdateOp,
::mlir::acc::WaitOp,
::mlir::acc::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace acc {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenACCOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::acc::ClauseDefaultValueAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: DefaultValue Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenACCOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::acc::ReductionOpAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: built-in reduction operations supported by OpenACC";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_OpenACCOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp definitions
//===----------------------------------------------------------------------===//

DataOpAdaptor::DataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.data", odsAttrs.getContext());
}

DataOpAdaptor::DataOpAdaptor(DataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DataOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DataOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange DataOpAdaptor::copyOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange DataOpAdaptor::copyinOperands() {
  return getODSOperands(2);
}

::mlir::ValueRange DataOpAdaptor::copyinReadonlyOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange DataOpAdaptor::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange DataOpAdaptor::copyoutZeroOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange DataOpAdaptor::createOperands() {
  return getODSOperands(6);
}

::mlir::ValueRange DataOpAdaptor::createZeroOperands() {
  return getODSOperands(7);
}

::mlir::ValueRange DataOpAdaptor::noCreateOperands() {
  return getODSOperands(8);
}

::mlir::ValueRange DataOpAdaptor::presentOperands() {
  return getODSOperands(9);
}

::mlir::ValueRange DataOpAdaptor::deviceptrOperands() {
  return getODSOperands(10);
}

::mlir::ValueRange DataOpAdaptor::attachOperands() {
  return getODSOperands(11);
}

::mlir::DictionaryAttr DataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::acc::ClauseDefaultValueAttr DataOpAdaptor::defaultAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DataOp::defaultAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
  return attr;
}

::llvm::Optional<::mlir::acc::ClauseDefaultValue> DataOpAdaptor::defaultAttr() {
  auto attr = defaultAttrAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange DataOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &DataOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult DataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DataOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DataOp::defaultAttrAttrName(*odsOpName)) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 12)
      return emitError(loc, "'acc.data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 12 "
                "elements, but got ") << numElements;
  }

  if (tblgen_defaultAttr && !((tblgen_defaultAttr.isa<::mlir::acc::ClauseDefaultValueAttr>())))
    return emitError(loc, "'acc.data' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> DataOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range DataOp::copyOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range DataOp::copyinOperands() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range DataOp::copyinReadonlyOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range DataOp::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range DataOp::copyoutZeroOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range DataOp::createOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range DataOp::createZeroOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range DataOp::noCreateOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range DataOp::presentOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range DataOp::deviceptrOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range DataOp::attachOperands() {
  return getODSOperands(11);
}

::mlir::MutableOperandRange DataOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::copyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::copyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::copyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::copyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::copyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::createOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::createZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::noCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::presentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::deviceptrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DataOp::attachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &DataOp::region() {
  return (*this)->getRegion(0);
}

::mlir::acc::ClauseDefaultValueAttr DataOp::defaultAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, defaultAttrAttrName()).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
}

::llvm::Optional<::mlir::acc::ClauseDefaultValue> DataOp::defaultAttr() {
  auto attr = defaultAttrAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::llvm::None);
}

void DataOp::defaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr) {
  (*this)->setAttr(defaultAttrAttrName(), attr);
}

::mlir::Attribute DataOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr(defaultAttrAttrName());
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(deviceptrOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(deviceptrOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (defaultAttr) {
  odsState.addAttribute(defaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void DataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(deviceptrOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(deviceptrOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (defaultAttr) {
  odsState.addAttribute(defaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == defaultAttrAttrName()) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 12)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 12 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_defaultAttr, "defaultAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult DataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyOperandsOperands;
  ::llvm::SMLoc copyOperandsOperandsLoc;
  (void)copyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinReadonlyOperandsOperands;
  ::llvm::SMLoc copyinReadonlyOperandsOperandsLoc;
  (void)copyinReadonlyOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinReadonlyOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutZeroOperandsOperands;
  ::llvm::SMLoc copyoutZeroOperandsOperandsLoc;
  (void)copyoutZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> noCreateOperandsOperands;
  ::llvm::SMLoc noCreateOperandsOperandsLoc;
  (void)noCreateOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> noCreateOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> presentOperandsOperands;
  ::llvm::SMLoc presentOperandsOperandsLoc;
  (void)presentOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> presentOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceptrOperandsOperands;
  ::llvm::SMLoc deviceptrOperandsOperandsLoc;
  (void)deviceptrOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceptrOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copy"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyin"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyin_readonly"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinReadonlyOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinReadonlyOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinReadonlyOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyout"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyout_zero"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create_zero"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("no_create"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  noCreateOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(noCreateOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(noCreateOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("present"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  presentOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(presentOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(presentOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("deviceptr"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceptrOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceptrOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceptrOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attach"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(copyOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(copyinReadonlyOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(copyoutZeroOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(noCreateOperandsOperands.size()), static_cast<int32_t>(presentOperandsOperands.size()), static_cast<int32_t>(deviceptrOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyOperandsOperands, copyOperandsTypes, copyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinReadonlyOperandsOperands, copyinReadonlyOperandsTypes, copyinReadonlyOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutZeroOperandsOperands, copyoutZeroOperandsTypes, copyoutZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(noCreateOperandsOperands, noCreateOperandsTypes, noCreateOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(presentOperandsOperands, presentOperandsTypes, presentOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceptrOperandsOperands, deviceptrOperandsTypes, deviceptrOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (!copyOperands().empty()) {
  _odsPrinter << ' ' << "copy";
  _odsPrinter << "(";
  _odsPrinter << copyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!copyinOperands().empty()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << copyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!copyinReadonlyOperands().empty()) {
  _odsPrinter << ' ' << "copyin_readonly";
  _odsPrinter << "(";
  _odsPrinter << copyinReadonlyOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyinReadonlyOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!copyoutOperands().empty()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << copyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!copyoutZeroOperands().empty()) {
  _odsPrinter << ' ' << "copyout_zero";
  _odsPrinter << "(";
  _odsPrinter << copyoutZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyoutZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!createOperands().empty()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << createOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << createOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!createZeroOperands().empty()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << createZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << createZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!noCreateOperands().empty()) {
  _odsPrinter << ' ' << "no_create";
  _odsPrinter << "(";
  _odsPrinter << noCreateOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << noCreateOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!presentOperands().empty()) {
  _odsPrinter << ' ' << "present";
  _odsPrinter << "(";
  _odsPrinter << presentOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << presentOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!deviceptrOperands().empty()) {
  _odsPrinter << ' ' << "deviceptr";
  _odsPrinter << "(";
  _odsPrinter << deviceptrOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << deviceptrOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!attachOperands().empty()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << attachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << attachOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::DataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp definitions
//===----------------------------------------------------------------------===//

EnterDataOpAdaptor::EnterDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.enter_data", odsAttrs.getContext());
}

EnterDataOpAdaptor::EnterDataOpAdaptor(EnterDataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange EnterDataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> EnterDataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, EnterDataOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange EnterDataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EnterDataOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange EnterDataOpAdaptor::waitOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange EnterDataOpAdaptor::copyinOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange EnterDataOpAdaptor::createOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange EnterDataOpAdaptor::createZeroOperands() {
  return getODSOperands(6);
}

::mlir::ValueRange EnterDataOpAdaptor::attachOperands() {
  return getODSOperands(7);
}

::mlir::DictionaryAttr EnterDataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr EnterDataOpAdaptor::asyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, EnterDataOp::asyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool EnterDataOpAdaptor::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr EnterDataOpAdaptor::waitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, EnterDataOp::waitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool EnterDataOpAdaptor::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::LogicalResult EnterDataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.enter_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == EnterDataOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == EnterDataOp::asyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == EnterDataOp::waitAttrName(*odsOpName)) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 8)
      return emitError(loc, "'acc.enter_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 8 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.enter_data' op ""attribute 'async' failed to satisfy constraint: unit attribute");

  if (tblgen_wait && !((tblgen_wait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.enter_data' op ""attribute 'wait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EnterDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range EnterDataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> EnterDataOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value EnterDataOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range EnterDataOp::waitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range EnterDataOp::copyinOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range EnterDataOp::createOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range EnterDataOp::createZeroOperands() {
  return getODSOperands(6);
}

::mlir::Operation::operand_range EnterDataOp::attachOperands() {
  return getODSOperands(7);
}

::mlir::MutableOperandRange EnterDataOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::copyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::createOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::createZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange EnterDataOp::attachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> EnterDataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EnterDataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr EnterDataOp::asyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, asyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr EnterDataOp::waitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, waitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool EnterDataOp::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

void EnterDataOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(asyncAttrName(), attr);
}

void EnterDataOp::waitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(waitAttrName(), attr);
}

::mlir::Attribute EnterDataOp::removeAsyncAttr() {
  return (*this)->removeAttr(asyncAttrName());
}

::mlir::Attribute EnterDataOp::removeWaitAttr() {
  return (*this)->removeAttr(waitAttrName());
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), wait);
  }
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), wait);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void EnterDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(attachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(attachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EnterDataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EnterDataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == asyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == waitAttrName()) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 8)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 8 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_async, "async")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_wait, "wait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EnterDataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult EnterDataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyinOperandsOperands;
  ::llvm::SMLoc copyinOperandsOperandsLoc;
  (void)copyinOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyinOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createOperandsOperands;
  ::llvm::SMLoc createOperandsOperandsLoc;
  (void)createOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> createZeroOperandsOperands;
  ::llvm::SMLoc createZeroOperandsOperandsLoc;
  (void)createZeroOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> createZeroOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> attachOperandsOperands;
  ::llvm::SMLoc attachOperandsOperandsLoc;
  (void)attachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> attachOperandsTypes;
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
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
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
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
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyin"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyinOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyinOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyinOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("create_zero"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  createZeroOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(createZeroOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(createZeroOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attach"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  attachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(attachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(attachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(copyinOperandsOperands.size()), static_cast<int32_t>(createOperandsOperands.size()), static_cast<int32_t>(createZeroOperandsOperands.size()), static_cast<int32_t>(attachOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyinOperandsOperands, copyinOperandsTypes, copyinOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createOperandsOperands, createOperandsTypes, createOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(createZeroOperandsOperands, createZeroOperandsTypes, createZeroOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(attachOperandsOperands, attachOperandsTypes, attachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EnterDataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (asyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = asyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (waitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = waitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (!waitOperands().empty()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << waitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << waitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!copyinOperands().empty()) {
  _odsPrinter << ' ' << "copyin";
  _odsPrinter << "(";
  _odsPrinter << copyinOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyinOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!createOperands().empty()) {
  _odsPrinter << ' ' << "create";
  _odsPrinter << "(";
  _odsPrinter << createOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << createOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!createZeroOperands().empty()) {
  _odsPrinter << ' ' << "create_zero";
  _odsPrinter << "(";
  _odsPrinter << createZeroOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << createZeroOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!attachOperands().empty()) {
  _odsPrinter << ' ' << "attach";
  _odsPrinter << "(";
  _odsPrinter << attachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << attachOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::EnterDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp definitions
//===----------------------------------------------------------------------===//

ExitDataOpAdaptor::ExitDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.exit_data", odsAttrs.getContext());
}

ExitDataOpAdaptor::ExitDataOpAdaptor(ExitDataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ExitDataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExitDataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExitDataOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ExitDataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExitDataOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ExitDataOpAdaptor::waitOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange ExitDataOpAdaptor::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange ExitDataOpAdaptor::deleteOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange ExitDataOpAdaptor::detachOperands() {
  return getODSOperands(6);
}

::mlir::DictionaryAttr ExitDataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ExitDataOpAdaptor::asyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ExitDataOp::asyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool ExitDataOpAdaptor::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOpAdaptor::waitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ExitDataOp::waitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool ExitDataOpAdaptor::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOpAdaptor::finalizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ExitDataOp::finalizeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool ExitDataOpAdaptor::finalize() {
  auto attr = finalizeAttr();
  return attr != nullptr;
}

::mlir::LogicalResult ExitDataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_finalize;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.exit_data' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ExitDataOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ExitDataOp::asyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ExitDataOp::finalizeAttrName(*odsOpName)) {
      tblgen_finalize = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ExitDataOp::waitAttrName(*odsOpName)) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'acc.exit_data' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.exit_data' op ""attribute 'async' failed to satisfy constraint: unit attribute");

  if (tblgen_wait && !((tblgen_wait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.exit_data' op ""attribute 'wait' failed to satisfy constraint: unit attribute");

  if (tblgen_finalize && !((tblgen_finalize.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.exit_data' op ""attribute 'finalize' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExitDataOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ExitDataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> ExitDataOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExitDataOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ExitDataOp::waitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ExitDataOp::copyoutOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range ExitDataOp::deleteOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range ExitDataOp::detachOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange ExitDataOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::copyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::deleteOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ExitDataOp::detachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ExitDataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExitDataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr ExitDataOp::asyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, asyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOp::waitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, waitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ExitDataOp::finalizeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, finalizeAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ExitDataOp::finalize() {
  auto attr = finalizeAttr();
  return attr != nullptr;
}

void ExitDataOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(asyncAttrName(), attr);
}

void ExitDataOp::waitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(waitAttrName(), attr);
}

void ExitDataOp::finalizeAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(finalizeAttrName(), attr);
}

::mlir::Attribute ExitDataOp::removeAsyncAttr() {
  return (*this)->removeAttr(asyncAttrName());
}

::mlir::Attribute ExitDataOp::removeWaitAttr() {
  return (*this)->removeAttr(waitAttrName());
}

::mlir::Attribute ExitDataOp::removeFinalizeAttr() {
  return (*this)->removeAttr(finalizeAttrName());
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), wait);
  }
  if (finalize) {
  odsState.addAttribute(finalizeAttrName(odsState.name), finalize);
  }
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), wait);
  }
  if (finalize) {
  odsState.addAttribute(finalizeAttrName(odsState.name), finalize);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (finalize) {
  odsState.addAttribute(finalizeAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void ExitDataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(deleteOperands);
  odsState.addOperands(detachOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(deleteOperands.size()), static_cast<int32_t>(detachOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (finalize) {
  odsState.addAttribute(finalizeAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExitDataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExitDataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_finalize;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == asyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == finalizeAttrName()) {
      tblgen_finalize = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == waitAttrName()) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_async, "async")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_wait, "wait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_finalize, "finalize")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExitDataOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExitDataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> copyoutOperandsOperands;
  ::llvm::SMLoc copyoutOperandsOperandsLoc;
  (void)copyoutOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> copyoutOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deleteOperandsOperands;
  ::llvm::SMLoc deleteOperandsOperandsLoc;
  (void)deleteOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deleteOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> detachOperandsOperands;
  ::llvm::SMLoc detachOperandsOperandsLoc;
  (void)detachOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> detachOperandsTypes;
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
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
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
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
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("copyout"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  copyoutOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(copyoutOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(copyoutOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("delete"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deleteOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deleteOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deleteOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("detach"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  detachOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(detachOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(detachOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(copyoutOperandsOperands.size()), static_cast<int32_t>(deleteOperandsOperands.size()), static_cast<int32_t>(detachOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(copyoutOperandsOperands, copyoutOperandsTypes, copyoutOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deleteOperandsOperands, deleteOperandsTypes, deleteOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(detachOperandsOperands, detachOperandsTypes, detachOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExitDataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (asyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = asyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (waitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = waitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (!waitOperands().empty()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << waitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << waitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!copyoutOperands().empty()) {
  _odsPrinter << ' ' << "copyout";
  _odsPrinter << "(";
  _odsPrinter << copyoutOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << copyoutOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!deleteOperands().empty()) {
  _odsPrinter << ' ' << "delete";
  _odsPrinter << "(";
  _odsPrinter << deleteOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << deleteOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!detachOperands().empty()) {
  _odsPrinter << ' ' << "detach";
  _odsPrinter << "(";
  _odsPrinter << detachOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << detachOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ExitDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp definitions
//===----------------------------------------------------------------------===//

InitOpAdaptor::InitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.init", odsAttrs.getContext());
}

InitOpAdaptor::InitOpAdaptor(InitOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange InitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InitOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange InitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InitOpAdaptor::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value InitOpAdaptor::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value InitOpAdaptor::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr InitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InitOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.init' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == InitOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.init' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> InitOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range InitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InitOp::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value InitOp::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::TypedValue<::mlir::IntegerType> InitOp::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange InitOp::deviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InitOp::deviceNumOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InitOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> InitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void InitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
}

void InitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InitOp::verifyInvariantsImpl() {
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InitOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceNumOperandOperands;
  ::llvm::SMLoc deviceNumOperandOperandsLoc;
  (void)deviceNumOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceNumOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (succeeded(parser.parseOptionalKeyword("device_type"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device_num"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceNumOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandOperands.push_back(operand);
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
      deviceNumOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(deviceNumOperandOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceNumOperandOperands, deviceNumOperandTypes, deviceNumOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!deviceTypeOperands().empty()) {
  _odsPrinter << ' ' << "device_type";
  _odsPrinter << "(";
  _odsPrinter << deviceTypeOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << deviceTypeOperands().getTypes();
  _odsPrinter << ")";
  }
  if (deviceNumOperand()) {
  _odsPrinter << ' ' << "device_num";
  _odsPrinter << "(";
  if (::mlir::Value value = deviceNumOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (deviceNumOperand() ? ::llvm::ArrayRef<::mlir::Type>(deviceNumOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::InitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp definitions
//===----------------------------------------------------------------------===//

LoopOpAdaptor::LoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.loop", odsAttrs.getContext());
}

LoopOpAdaptor::LoopOpAdaptor(LoopOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LoopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange LoopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopOpAdaptor::gangNum() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOpAdaptor::gangStatic() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOpAdaptor::workerNum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOpAdaptor::vectorLength() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange LoopOpAdaptor::tileOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange LoopOpAdaptor::privateOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange LoopOpAdaptor::reductionOperands() {
  return getODSOperands(6);
}

::mlir::DictionaryAttr LoopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopOpAdaptor::collapseAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::collapseAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint64_t> LoopOpAdaptor::collapse() {
  auto attr = collapseAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr LoopOpAdaptor::seqAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LoopOp::seqAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool LoopOpAdaptor::seq() {
  auto attr = seqAttr();
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpAdaptor::independentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::independentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool LoopOpAdaptor::independent() {
  auto attr = independentAttr();
  return attr != nullptr;
}

::mlir::UnitAttr LoopOpAdaptor::auto_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::auto_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool LoopOpAdaptor::auto_() {
  auto attr = auto_Attr();
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr LoopOpAdaptor::reductionOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LoopOp::reductionOpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
  return attr;
}

::llvm::Optional<::mlir::acc::ReductionOp> LoopOpAdaptor::reductionOp() {
  auto attr = reductionOpAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr LoopOpAdaptor::exec_mappingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LoopOp::exec_mappingAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopOpAdaptor::exec_mapping() {
  auto attr = exec_mappingAttr();
  return attr.getValue().getZExtValue();
}

::mlir::RegionRange LoopOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &LoopOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult LoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_auto_;
  ::mlir::Attribute tblgen_collapse;
  ::mlir::Attribute tblgen_exec_mapping;
  ::mlir::Attribute tblgen_independent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.loop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == LoopOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == LoopOp::auto_AttrName(*odsOpName)) {
      tblgen_auto_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::collapseAttrName(*odsOpName)) {
      tblgen_collapse = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::exec_mappingAttrName(*odsOpName)) {
      tblgen_exec_mapping = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::independentAttrName(*odsOpName)) {
      tblgen_independent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_seq;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoopOp::reductionOpAttrName(*odsOpName)) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopOp::seqAttrName(*odsOpName)) {
      tblgen_seq = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'acc.loop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_collapse && !(((tblgen_collapse.isa<::mlir::IntegerAttr>())) && ((tblgen_collapse.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'acc.loop' op ""attribute 'collapse' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_seq && !((tblgen_seq.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'seq' failed to satisfy constraint: unit attribute");

  if (tblgen_independent && !((tblgen_independent.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'independent' failed to satisfy constraint: unit attribute");

  if (tblgen_auto_ && !((tblgen_auto_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'auto_' failed to satisfy constraint: unit attribute");

  if (tblgen_reductionOp && !((tblgen_reductionOp.isa<::mlir::acc::ReductionOpAttr>())))
    return emitError(loc, "'acc.loop' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");

  if (tblgen_exec_mapping && !(((tblgen_exec_mapping.isa<::mlir::IntegerAttr>())) && ((tblgen_exec_mapping.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'acc.loop' op ""attribute 'exec_mapping' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range LoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopOp::gangNum() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOp::gangStatic() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOp::workerNum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value LoopOp::vectorLength() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range LoopOp::tileOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range LoopOp::privateOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range LoopOp::reductionOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange LoopOp::gangNumMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::gangStaticMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::workerNumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::vectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::tileOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::privateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange LoopOp::reductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range LoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range LoopOp::results() {
  return getODSResults(0);
}

::mlir::Region &LoopOp::region() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr LoopOp::collapseAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, collapseAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> LoopOp::collapse() {
  auto attr = collapseAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr LoopOp::seqAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, seqAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::seq() {
  auto attr = seqAttr();
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::independentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, independentAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::independent() {
  auto attr = independentAttr();
  return attr != nullptr;
}

::mlir::UnitAttr LoopOp::auto_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, auto_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoopOp::auto_() {
  auto attr = auto_Attr();
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr LoopOp::reductionOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, reductionOpAttrName()).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
}

::llvm::Optional<::mlir::acc::ReductionOp> LoopOp::reductionOp() {
  auto attr = reductionOpAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr LoopOp::exec_mappingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, exec_mappingAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t LoopOp::exec_mapping() {
  auto attr = exec_mappingAttr();
  return attr.getValue().getZExtValue();
}

void LoopOp::collapseAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(collapseAttrName(), attr);
}

void LoopOp::seqAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(seqAttrName(), attr);
}

void LoopOp::independentAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(independentAttrName(), attr);
}

void LoopOp::auto_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(auto_AttrName(), attr);
}

void LoopOp::reductionOpAttr(::mlir::acc::ReductionOpAttr attr) {
  (*this)->setAttr(reductionOpAttrName(), attr);
}

void LoopOp::exec_mappingAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(exec_mappingAttrName(), attr);
}

::mlir::Attribute LoopOp::removeCollapseAttr() {
  return (*this)->removeAttr(collapseAttrName());
}

::mlir::Attribute LoopOp::removeSeqAttr() {
  return (*this)->removeAttr(seqAttrName());
}

::mlir::Attribute LoopOp::removeIndependentAttr() {
  return (*this)->removeAttr(independentAttrName());
}

::mlir::Attribute LoopOp::removeAuto_Attr() {
  return (*this)->removeAttr(auto_AttrName());
}

::mlir::Attribute LoopOp::removeReductionOpAttr() {
  return (*this)->removeAttr(reductionOpAttrName());
}

void LoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::IntegerAttr exec_mapping) {
  if (gangNum)
    odsState.addOperands(gangNum);
  if (gangStatic)
    odsState.addOperands(gangStatic);
  if (workerNum)
    odsState.addOperands(workerNum);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  odsState.addOperands(tileOperands);
  odsState.addOperands(privateOperands);
  odsState.addOperands(reductionOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(gangNum ? 1 : 0), (gangStatic ? 1 : 0), (workerNum ? 1 : 0), (vectorLength ? 1 : 0), static_cast<int32_t>(tileOperands.size()), static_cast<int32_t>(privateOperands.size()), static_cast<int32_t>(reductionOperands.size())}));
  if (collapse) {
  odsState.addAttribute(collapseAttrName(odsState.name), collapse);
  }
  if (seq) {
  odsState.addAttribute(seqAttrName(odsState.name), seq);
  }
  if (independent) {
  odsState.addAttribute(independentAttrName(odsState.name), independent);
  }
  if (auto_) {
  odsState.addAttribute(auto_AttrName(odsState.name), auto_);
  }
  if (reductionOp) {
  odsState.addAttribute(reductionOpAttrName(odsState.name), reductionOp);
  }
  if (exec_mapping) {
  odsState.addAttribute(exec_mappingAttrName(odsState.name), exec_mapping);
  }
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, uint64_t exec_mapping) {
  if (gangNum)
    odsState.addOperands(gangNum);
  if (gangStatic)
    odsState.addOperands(gangStatic);
  if (workerNum)
    odsState.addOperands(workerNum);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  odsState.addOperands(tileOperands);
  odsState.addOperands(privateOperands);
  odsState.addOperands(reductionOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(gangNum ? 1 : 0), (gangStatic ? 1 : 0), (workerNum ? 1 : 0), (vectorLength ? 1 : 0), static_cast<int32_t>(tileOperands.size()), static_cast<int32_t>(privateOperands.size()), static_cast<int32_t>(reductionOperands.size())}));
  if (collapse) {
  odsState.addAttribute(collapseAttrName(odsState.name), collapse);
  }
  if (seq) {
  odsState.addAttribute(seqAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (independent) {
  odsState.addAttribute(independentAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (auto_) {
  odsState.addAttribute(auto_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (reductionOp) {
  odsState.addAttribute(reductionOpAttrName(odsState.name), reductionOp);
  }
  odsState.addAttribute(exec_mappingAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), exec_mapping));
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void LoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

void LoopOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[2])) {
     attributes.append(attrNames[2], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 0));
  }
}

::mlir::LogicalResult LoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_auto_;
  ::mlir::Attribute tblgen_collapse;
  ::mlir::Attribute tblgen_exec_mapping;
  ::mlir::Attribute tblgen_independent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == auto_AttrName()) {
      tblgen_auto_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == collapseAttrName()) {
      tblgen_collapse = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == exec_mappingAttrName()) {
      tblgen_exec_mapping = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == independentAttrName()) {
      tblgen_independent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_seq;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == reductionOpAttrName()) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == seqAttrName()) {
      tblgen_seq = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_collapse, "collapse")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_seq, "seq")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_independent, "independent")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_auto_, "auto_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps3(*this, tblgen_reductionOp, "reductionOp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps2(*this, tblgen_exec_mapping, "exec_mapping")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::LoopOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp definitions
//===----------------------------------------------------------------------===//

ParallelOpAdaptor::ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.parallel", odsAttrs.getContext());
}

ParallelOpAdaptor::ParallelOpAdaptor(ParallelOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ParallelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ParallelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ParallelOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ParallelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOpAdaptor::async() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ParallelOpAdaptor::waitOperands() {
  return getODSOperands(1);
}

::mlir::Value ParallelOpAdaptor::numGangs() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::numWorkers() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::vectorLength() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::ifCond() {
  auto operands = getODSOperands(5);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::selfCond() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ParallelOpAdaptor::reductionOperands() {
  return getODSOperands(7);
}

::mlir::ValueRange ParallelOpAdaptor::copyOperands() {
  return getODSOperands(8);
}

::mlir::ValueRange ParallelOpAdaptor::copyinOperands() {
  return getODSOperands(9);
}

::mlir::ValueRange ParallelOpAdaptor::copyinReadonlyOperands() {
  return getODSOperands(10);
}

::mlir::ValueRange ParallelOpAdaptor::copyoutOperands() {
  return getODSOperands(11);
}

::mlir::ValueRange ParallelOpAdaptor::copyoutZeroOperands() {
  return getODSOperands(12);
}

::mlir::ValueRange ParallelOpAdaptor::createOperands() {
  return getODSOperands(13);
}

::mlir::ValueRange ParallelOpAdaptor::createZeroOperands() {
  return getODSOperands(14);
}

::mlir::ValueRange ParallelOpAdaptor::noCreateOperands() {
  return getODSOperands(15);
}

::mlir::ValueRange ParallelOpAdaptor::presentOperands() {
  return getODSOperands(16);
}

::mlir::ValueRange ParallelOpAdaptor::devicePtrOperands() {
  return getODSOperands(17);
}

::mlir::ValueRange ParallelOpAdaptor::attachOperands() {
  return getODSOperands(18);
}

::mlir::ValueRange ParallelOpAdaptor::gangPrivateOperands() {
  return getODSOperands(19);
}

::mlir::ValueRange ParallelOpAdaptor::gangFirstPrivateOperands() {
  return getODSOperands(20);
}

::mlir::DictionaryAttr ParallelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr ParallelOpAdaptor::asyncAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ParallelOp::asyncAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool ParallelOpAdaptor::asyncAttr() {
  auto attr = asyncAttrAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOpAdaptor::waitAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::waitAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool ParallelOpAdaptor::waitAttr() {
  auto attr = waitAttrAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOpAdaptor::selfAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::selfAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool ParallelOpAdaptor::selfAttr() {
  auto attr = selfAttrAttr();
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr ParallelOpAdaptor::reductionOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::reductionOpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
  return attr;
}

::llvm::Optional<::mlir::acc::ReductionOp> ParallelOpAdaptor::reductionOp() {
  auto attr = reductionOpAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::llvm::None);
}

::mlir::acc::ClauseDefaultValueAttr ParallelOpAdaptor::defaultAttrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ParallelOp::defaultAttrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
  return attr;
}

::llvm::Optional<::mlir::acc::ClauseDefaultValue> ParallelOpAdaptor::defaultAttr() {
  auto attr = defaultAttrAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange ParallelOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ParallelOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ParallelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.parallel' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ParallelOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ParallelOp::asyncAttrAttrName(*odsOpName)) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::defaultAttrAttrName(*odsOpName)) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ParallelOp::reductionOpAttrName(*odsOpName)) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::selfAttrAttrName(*odsOpName)) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::waitAttrAttrName(*odsOpName)) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 21)
      return emitError(loc, "'acc.parallel' op ""'operand_segment_sizes' attribute for specifying operand segments must have 21 "
                "elements, but got ") << numElements;
  }

  if (tblgen_asyncAttr && !((tblgen_asyncAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'asyncAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_waitAttr && !((tblgen_waitAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'waitAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_selfAttr && !((tblgen_selfAttr.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'selfAttr' failed to satisfy constraint: unit attribute");

  if (tblgen_reductionOp && !((tblgen_reductionOp.isa<::mlir::acc::ReductionOpAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'reductionOp' failed to satisfy constraint: built-in reduction operations supported by OpenACC");

  if (tblgen_defaultAttr && !((tblgen_defaultAttr.isa<::mlir::acc::ClauseDefaultValueAttr>())))
    return emitError(loc, "'acc.parallel' op ""attribute 'defaultAttr' failed to satisfy constraint: DefaultValue Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ParallelOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ParallelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOp::async() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ParallelOp::waitOperands() {
  return getODSOperands(1);
}

::mlir::Value ParallelOp::numGangs() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::numWorkers() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::vectorLength() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::TypedValue<::mlir::IntegerType> ParallelOp::ifCond() {
  auto operands = getODSOperands(5);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::TypedValue<::mlir::IntegerType> ParallelOp::selfCond() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ParallelOp::reductionOperands() {
  return getODSOperands(7);
}

::mlir::Operation::operand_range ParallelOp::copyOperands() {
  return getODSOperands(8);
}

::mlir::Operation::operand_range ParallelOp::copyinOperands() {
  return getODSOperands(9);
}

::mlir::Operation::operand_range ParallelOp::copyinReadonlyOperands() {
  return getODSOperands(10);
}

::mlir::Operation::operand_range ParallelOp::copyoutOperands() {
  return getODSOperands(11);
}

::mlir::Operation::operand_range ParallelOp::copyoutZeroOperands() {
  return getODSOperands(12);
}

::mlir::Operation::operand_range ParallelOp::createOperands() {
  return getODSOperands(13);
}

::mlir::Operation::operand_range ParallelOp::createZeroOperands() {
  return getODSOperands(14);
}

::mlir::Operation::operand_range ParallelOp::noCreateOperands() {
  return getODSOperands(15);
}

::mlir::Operation::operand_range ParallelOp::presentOperands() {
  return getODSOperands(16);
}

::mlir::Operation::operand_range ParallelOp::devicePtrOperands() {
  return getODSOperands(17);
}

::mlir::Operation::operand_range ParallelOp::attachOperands() {
  return getODSOperands(18);
}

::mlir::Operation::operand_range ParallelOp::gangPrivateOperands() {
  return getODSOperands(19);
}

::mlir::Operation::operand_range ParallelOp::gangFirstPrivateOperands() {
  return getODSOperands(20);
}

::mlir::MutableOperandRange ParallelOp::asyncMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::numGangsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::numWorkersMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::vectorLengthMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::selfCondMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::reductionOperandsMutable() {
  auto range = getODSOperandIndexAndLength(7);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(7u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::copyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(8);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(8u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::copyinOperandsMutable() {
  auto range = getODSOperandIndexAndLength(9);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(9u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::copyinReadonlyOperandsMutable() {
  auto range = getODSOperandIndexAndLength(10);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(10u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::copyoutOperandsMutable() {
  auto range = getODSOperandIndexAndLength(11);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(11u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::copyoutZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(12);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(12u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::createOperandsMutable() {
  auto range = getODSOperandIndexAndLength(13);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(13u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::createZeroOperandsMutable() {
  auto range = getODSOperandIndexAndLength(14);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(14u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::noCreateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(15);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(15u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::presentOperandsMutable() {
  auto range = getODSOperandIndexAndLength(16);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(16u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::devicePtrOperandsMutable() {
  auto range = getODSOperandIndexAndLength(17);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(17u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::attachOperandsMutable() {
  auto range = getODSOperandIndexAndLength(18);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(18u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::gangPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(19);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(19u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::gangFirstPrivateOperandsMutable() {
  auto range = getODSOperandIndexAndLength(20);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(20u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ParallelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ParallelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ParallelOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr ParallelOp::asyncAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, asyncAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::asyncAttr() {
  auto attr = asyncAttrAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOp::waitAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, waitAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::waitAttr() {
  auto attr = waitAttrAttr();
  return attr != nullptr;
}

::mlir::UnitAttr ParallelOp::selfAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, selfAttrAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool ParallelOp::selfAttr() {
  auto attr = selfAttrAttr();
  return attr != nullptr;
}

::mlir::acc::ReductionOpAttr ParallelOp::reductionOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, reductionOpAttrName()).dyn_cast_or_null<::mlir::acc::ReductionOpAttr>();
}

::llvm::Optional<::mlir::acc::ReductionOp> ParallelOp::reductionOp() {
  auto attr = reductionOpAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ReductionOp>(attr.getValue()) : (::llvm::None);
}

::mlir::acc::ClauseDefaultValueAttr ParallelOp::defaultAttrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, defaultAttrAttrName()).dyn_cast_or_null<::mlir::acc::ClauseDefaultValueAttr>();
}

::llvm::Optional<::mlir::acc::ClauseDefaultValue> ParallelOp::defaultAttr() {
  auto attr = defaultAttrAttr();
  return attr ? ::llvm::Optional<::mlir::acc::ClauseDefaultValue>(attr.getValue()) : (::llvm::None);
}

void ParallelOp::asyncAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(asyncAttrAttrName(), attr);
}

void ParallelOp::waitAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(waitAttrAttrName(), attr);
}

void ParallelOp::selfAttrAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(selfAttrAttrName(), attr);
}

void ParallelOp::reductionOpAttr(::mlir::acc::ReductionOpAttr attr) {
  (*this)->setAttr(reductionOpAttrName(), attr);
}

void ParallelOp::defaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr) {
  (*this)->setAttr(defaultAttrAttrName(), attr);
}

::mlir::Attribute ParallelOp::removeAsyncAttrAttr() {
  return (*this)->removeAttr(asyncAttrAttrName());
}

::mlir::Attribute ParallelOp::removeWaitAttrAttr() {
  return (*this)->removeAttr(waitAttrAttrName());
}

::mlir::Attribute ParallelOp::removeSelfAttrAttr() {
  return (*this)->removeAttr(selfAttrAttrName());
}

::mlir::Attribute ParallelOp::removeReductionOpAttr() {
  return (*this)->removeAttr(reductionOpAttrName());
}

::mlir::Attribute ParallelOp::removeDefaultAttrAttr() {
  return (*this)->removeAttr(defaultAttrAttrName());
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute(asyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
  odsState.addAttribute(waitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
  odsState.addAttribute(selfAttrAttrName(odsState.name), selfAttr);
  }
  if (reductionOp) {
  odsState.addAttribute(reductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute(defaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute(asyncAttrAttrName(odsState.name), asyncAttr);
  }
  if (waitAttr) {
  odsState.addAttribute(waitAttrAttrName(odsState.name), waitAttr);
  }
  if (selfAttr) {
  odsState.addAttribute(selfAttrAttrName(odsState.name), selfAttr);
  }
  if (reductionOp) {
  odsState.addAttribute(reductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute(defaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute(asyncAttrAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (waitAttr) {
  odsState.addAttribute(waitAttrAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (selfAttr) {
  odsState.addAttribute(selfAttrAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (reductionOp) {
  odsState.addAttribute(reductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute(defaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr) {
  if (async)
    odsState.addOperands(async);
  odsState.addOperands(waitOperands);
  if (numGangs)
    odsState.addOperands(numGangs);
  if (numWorkers)
    odsState.addOperands(numWorkers);
  if (vectorLength)
    odsState.addOperands(vectorLength);
  if (ifCond)
    odsState.addOperands(ifCond);
  if (selfCond)
    odsState.addOperands(selfCond);
  odsState.addOperands(reductionOperands);
  odsState.addOperands(copyOperands);
  odsState.addOperands(copyinOperands);
  odsState.addOperands(copyinReadonlyOperands);
  odsState.addOperands(copyoutOperands);
  odsState.addOperands(copyoutZeroOperands);
  odsState.addOperands(createOperands);
  odsState.addOperands(createZeroOperands);
  odsState.addOperands(noCreateOperands);
  odsState.addOperands(presentOperands);
  odsState.addOperands(devicePtrOperands);
  odsState.addOperands(attachOperands);
  odsState.addOperands(gangPrivateOperands);
  odsState.addOperands(gangFirstPrivateOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(async ? 1 : 0), static_cast<int32_t>(waitOperands.size()), (numGangs ? 1 : 0), (numWorkers ? 1 : 0), (vectorLength ? 1 : 0), (ifCond ? 1 : 0), (selfCond ? 1 : 0), static_cast<int32_t>(reductionOperands.size()), static_cast<int32_t>(copyOperands.size()), static_cast<int32_t>(copyinOperands.size()), static_cast<int32_t>(copyinReadonlyOperands.size()), static_cast<int32_t>(copyoutOperands.size()), static_cast<int32_t>(copyoutZeroOperands.size()), static_cast<int32_t>(createOperands.size()), static_cast<int32_t>(createZeroOperands.size()), static_cast<int32_t>(noCreateOperands.size()), static_cast<int32_t>(presentOperands.size()), static_cast<int32_t>(devicePtrOperands.size()), static_cast<int32_t>(attachOperands.size()), static_cast<int32_t>(gangPrivateOperands.size()), static_cast<int32_t>(gangFirstPrivateOperands.size())}));
  if (asyncAttr) {
  odsState.addAttribute(asyncAttrAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (waitAttr) {
  odsState.addAttribute(waitAttrAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (selfAttr) {
  odsState.addAttribute(selfAttrAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (reductionOp) {
  odsState.addAttribute(reductionOpAttrName(odsState.name), reductionOp);
  }
  if (defaultAttr) {
  odsState.addAttribute(defaultAttrAttrName(odsState.name), defaultAttr);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ParallelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_asyncAttr;
  ::mlir::Attribute tblgen_defaultAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == asyncAttrAttrName()) {
      tblgen_asyncAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == defaultAttrAttrName()) {
      tblgen_defaultAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductionOp;
  ::mlir::Attribute tblgen_selfAttr;
  ::mlir::Attribute tblgen_waitAttr;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == reductionOpAttrName()) {
      tblgen_reductionOp = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == selfAttrAttrName()) {
      tblgen_selfAttr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == waitAttrAttrName()) {
      tblgen_waitAttr = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 21)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 21 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_asyncAttr, "asyncAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_waitAttr, "waitAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_selfAttr, "selfAttr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps3(*this, tblgen_reductionOp, "reductionOp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps0(*this, tblgen_defaultAttr, "defaultAttr")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    if (valueGroup5.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup5.size();
    }

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    if (valueGroup6.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup6.size();
    }

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup7 = getODSOperands(7);

    for (auto v : valueGroup7) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup8 = getODSOperands(8);

    for (auto v : valueGroup8) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup9 = getODSOperands(9);

    for (auto v : valueGroup9) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup10 = getODSOperands(10);

    for (auto v : valueGroup10) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup11 = getODSOperands(11);

    for (auto v : valueGroup11) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup12 = getODSOperands(12);

    for (auto v : valueGroup12) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup13 = getODSOperands(13);

    for (auto v : valueGroup13) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup14 = getODSOperands(14);

    for (auto v : valueGroup14) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup15 = getODSOperands(15);

    for (auto v : valueGroup15) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup16 = getODSOperands(16);

    for (auto v : valueGroup16) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup17 = getODSOperands(17);

    for (auto v : valueGroup17) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup18 = getODSOperands(18);

    for (auto v : valueGroup18) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup19 = getODSOperands(19);

    for (auto v : valueGroup19) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup20 = getODSOperands(20);

    for (auto v : valueGroup20) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenACCOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ParallelOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ParallelOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp definitions
//===----------------------------------------------------------------------===//

ShutdownOpAdaptor::ShutdownOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.shutdown", odsAttrs.getContext());
}

ShutdownOpAdaptor::ShutdownOpAdaptor(ShutdownOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ShutdownOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShutdownOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ShutdownOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ShutdownOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ShutdownOpAdaptor::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value ShutdownOpAdaptor::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ShutdownOpAdaptor::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr ShutdownOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ShutdownOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.shutdown' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ShutdownOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'acc.shutdown' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> ShutdownOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ShutdownOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ShutdownOp::deviceTypeOperands() {
  return getODSOperands(0);
}

::mlir::Value ShutdownOp::deviceNumOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::TypedValue<::mlir::IntegerType> ShutdownOp::ifCond() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange ShutdownOp::deviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ShutdownOp::deviceNumOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ShutdownOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ShutdownOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShutdownOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ShutdownOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
}

void ShutdownOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(deviceTypeOperands);
  if (deviceNumOperand)
    odsState.addOperands(deviceNumOperand);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperands.size()), (deviceNumOperand ? 1 : 0), (ifCond ? 1 : 0)}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShutdownOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShutdownOp::verifyInvariantsImpl() {
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShutdownOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ShutdownOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceNumOperandOperands;
  ::llvm::SMLoc deviceNumOperandOperandsLoc;
  (void)deviceNumOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceNumOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (succeeded(parser.parseOptionalKeyword("device_type"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device_num"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceNumOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceNumOperandOperands.push_back(operand);
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
      deviceNumOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(deviceNumOperandOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceNumOperandOperands, deviceNumOperandTypes, deviceNumOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShutdownOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!deviceTypeOperands().empty()) {
  _odsPrinter << ' ' << "device_type";
  _odsPrinter << "(";
  _odsPrinter << deviceTypeOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << deviceTypeOperands().getTypes();
  _odsPrinter << ")";
  }
  if (deviceNumOperand()) {
  _odsPrinter << ' ' << "device_num";
  _odsPrinter << "(";
  if (::mlir::Value value = deviceNumOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (deviceNumOperand() ? ::llvm::ArrayRef<::mlir::Type>(deviceNumOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::ShutdownOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp definitions
//===----------------------------------------------------------------------===//

TerminatorOpAdaptor::TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.terminator", odsAttrs.getContext());
}

TerminatorOpAdaptor::TerminatorOpAdaptor(TerminatorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TerminatorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TerminatorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TerminatorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TerminatorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TerminatorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TerminatorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TerminatorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TerminatorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TerminatorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TerminatorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TerminatorOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TerminatorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TerminatorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TerminatorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::TerminatorOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp definitions
//===----------------------------------------------------------------------===//

UpdateOpAdaptor::UpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.update", odsAttrs.getContext());
}

UpdateOpAdaptor::UpdateOpAdaptor(UpdateOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UpdateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UpdateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, UpdateOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange UpdateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UpdateOpAdaptor::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value UpdateOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value UpdateOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange UpdateOpAdaptor::waitOperands() {
  return getODSOperands(3);
}

::mlir::ValueRange UpdateOpAdaptor::deviceTypeOperands() {
  return getODSOperands(4);
}

::mlir::ValueRange UpdateOpAdaptor::hostOperands() {
  return getODSOperands(5);
}

::mlir::ValueRange UpdateOpAdaptor::deviceOperands() {
  return getODSOperands(6);
}

::mlir::DictionaryAttr UpdateOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr UpdateOpAdaptor::asyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, UpdateOp::asyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool UpdateOpAdaptor::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOpAdaptor::waitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, UpdateOp::waitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool UpdateOpAdaptor::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOpAdaptor::ifPresentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, UpdateOp::ifPresentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool UpdateOpAdaptor::ifPresent() {
  auto attr = ifPresentAttr();
  return attr != nullptr;
}

::mlir::LogicalResult UpdateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_ifPresent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.update' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == UpdateOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == UpdateOp::asyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == UpdateOp::ifPresentAttrName(*odsOpName)) {
      tblgen_ifPresent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == UpdateOp::waitAttrName(*odsOpName)) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitError(loc, "'acc.update' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.update' op ""attribute 'async' failed to satisfy constraint: unit attribute");

  if (tblgen_wait && !((tblgen_wait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.update' op ""attribute 'wait' failed to satisfy constraint: unit attribute");

  if (tblgen_ifPresent && !((tblgen_ifPresent.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.update' op ""attribute 'ifPresent' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UpdateOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range UpdateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IntegerType> UpdateOp::ifCond() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value UpdateOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value UpdateOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range UpdateOp::waitOperands() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range UpdateOp::deviceTypeOperands() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range UpdateOp::hostOperands() {
  return getODSOperands(5);
}

::mlir::Operation::operand_range UpdateOp::deviceOperands() {
  return getODSOperands(6);
}

::mlir::MutableOperandRange UpdateOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::deviceTypeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::hostOperandsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange UpdateOp::deviceOperandsMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> UpdateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UpdateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr UpdateOp::asyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, asyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOp::waitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, waitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::wait() {
  auto attr = waitAttr();
  return attr != nullptr;
}

::mlir::UnitAttr UpdateOp::ifPresentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, ifPresentAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool UpdateOp::ifPresent() {
  auto attr = ifPresentAttr();
  return attr != nullptr;
}

void UpdateOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(asyncAttrName(), attr);
}

void UpdateOp::waitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(waitAttrName(), attr);
}

void UpdateOp::ifPresentAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(ifPresentAttrName(), attr);
}

::mlir::Attribute UpdateOp::removeAsyncAttr() {
  return (*this)->removeAttr(asyncAttrName());
}

::mlir::Attribute UpdateOp::removeWaitAttr() {
  return (*this)->removeAttr(waitAttrName());
}

::mlir::Attribute UpdateOp::removeIfPresentAttr() {
  return (*this)->removeAttr(ifPresentAttrName());
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), wait);
  }
  if (ifPresent) {
  odsState.addAttribute(ifPresentAttrName(odsState.name), ifPresent);
  }
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), wait);
  }
  if (ifPresent) {
  odsState.addAttribute(ifPresentAttrName(odsState.name), ifPresent);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (ifPresent) {
  odsState.addAttribute(ifPresentAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void UpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent) {
  if (ifCond)
    odsState.addOperands(ifCond);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  odsState.addOperands(waitOperands);
  odsState.addOperands(deviceTypeOperands);
  odsState.addOperands(hostOperands);
  odsState.addOperands(deviceOperands);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({(ifCond ? 1 : 0), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), static_cast<int32_t>(waitOperands.size()), static_cast<int32_t>(deviceTypeOperands.size()), static_cast<int32_t>(hostOperands.size()), static_cast<int32_t>(deviceOperands.size())}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (wait) {
  odsState.addAttribute(waitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (ifPresent) {
  odsState.addAttribute(ifPresentAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UpdateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  ::mlir::Attribute tblgen_ifPresent;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == asyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ifPresentAttrName()) {
      tblgen_ifPresent = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_wait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == waitAttrName()) {
      tblgen_wait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_async, "async")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_wait, "wait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_ifPresent, "ifPresent")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UpdateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult UpdateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceTypeOperandsOperands;
  ::llvm::SMLoc deviceTypeOperandsOperandsLoc;
  (void)deviceTypeOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypeOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> hostOperandsOperands;
  ::llvm::SMLoc hostOperandsOperandsLoc;
  (void)hostOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> hostOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperandsOperands;
  ::llvm::SMLoc deviceOperandsOperandsLoc;
  (void)deviceOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceOperandsTypes;
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
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
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
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
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device_type"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceTypeOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceTypeOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceTypeOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("host"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  hostOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(hostOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(hostOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("device"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  deviceOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(deviceOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(deviceOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ifCondOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(deviceTypeOperandsOperands.size()), static_cast<int32_t>(hostOperandsOperands.size()), static_cast<int32_t>(deviceOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceTypeOperandsOperands, deviceTypeOperandsTypes, deviceTypeOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(hostOperandsOperands, hostOperandsTypes, hostOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperandsOperands, deviceOperandsTypes, deviceOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UpdateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (asyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = asyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (waitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = waitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (!deviceTypeOperands().empty()) {
  _odsPrinter << ' ' << "device_type";
  _odsPrinter << "(";
  _odsPrinter << deviceTypeOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << deviceTypeOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!waitOperands().empty()) {
  _odsPrinter << ' ' << "wait";
  _odsPrinter << "(";
  _odsPrinter << waitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << waitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!hostOperands().empty()) {
  _odsPrinter << ' ' << "host";
  _odsPrinter << "(";
  _odsPrinter << hostOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << hostOperands().getTypes();
  _odsPrinter << ")";
  }
  if (!deviceOperands().empty()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  _odsPrinter << deviceOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << deviceOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::UpdateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp definitions
//===----------------------------------------------------------------------===//

WaitOpAdaptor::WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.wait", odsAttrs.getContext());
}

WaitOpAdaptor::WaitOpAdaptor(WaitOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WaitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WaitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, WaitOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange WaitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WaitOpAdaptor::waitOperands() {
  return getODSOperands(0);
}

::mlir::Value WaitOpAdaptor::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOpAdaptor::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOpAdaptor::ifCond() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr WaitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr WaitOpAdaptor::asyncAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WaitOp::asyncAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool WaitOpAdaptor::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

::mlir::LogicalResult WaitOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'acc.wait' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == WaitOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WaitOp::asyncAttrName(*odsOpName)) {
      tblgen_async = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'acc.wait' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_async && !((tblgen_async.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'acc.wait' op ""attribute 'async' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WaitOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range WaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WaitOp::waitOperands() {
  return getODSOperands(0);
}

::mlir::Value WaitOp::asyncOperand() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value WaitOp::waitDevnum() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::TypedValue<::mlir::IntegerType> WaitOp::ifCond() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange WaitOp::waitOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WaitOp::asyncOperandMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WaitOp::waitDevnumMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WaitOp::ifCondMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> WaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr WaitOp::asyncAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, asyncAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WaitOp::async() {
  auto attr = asyncAttr();
  return attr != nullptr;
}

void WaitOp::asyncAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(asyncAttrName(), attr);
}

::mlir::Attribute WaitOp::removeAsyncAttr() {
  return (*this)->removeAttr(asyncAttrName());
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), async);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void WaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond) {
  odsState.addOperands(waitOperands);
  if (asyncOperand)
    odsState.addOperands(asyncOperand);
  if (waitDevnum)
    odsState.addOperands(waitDevnum);
  if (ifCond)
    odsState.addOperands(ifCond);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(waitOperands.size()), (asyncOperand ? 1 : 0), (waitDevnum ? 1 : 0), (ifCond ? 1 : 0)}));
  if (async) {
  odsState.addAttribute(asyncAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WaitOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_async;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == asyncAttrName()) {
      tblgen_async = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenACCOps1(*this, tblgen_async, "async")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WaitOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitOperandsOperands;
  ::llvm::SMLoc waitOperandsOperandsLoc;
  (void)waitOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> asyncOperandOperands;
  ::llvm::SMLoc asyncOperandOperandsLoc;
  (void)asyncOperandOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> asyncOperandTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> waitDevnumOperands;
  ::llvm::SMLoc waitDevnumOperandsLoc;
  (void)waitDevnumOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> waitDevnumTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ifCondOperands;
  ::llvm::SMLoc ifCondOperandsLoc;
  (void)ifCondOperandsLoc;
  if (succeeded(parser.parseOptionalLParen())) {

  waitOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(waitOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(waitOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("async"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    asyncOperandOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      asyncOperandOperands.push_back(operand);
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
      asyncOperandTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("wait_devnum"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    waitDevnumOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      waitDevnumOperands.push_back(operand);
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
      waitDevnumTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ifCondOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      ifCondOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(waitOperandsOperands.size()), static_cast<int32_t>(asyncOperandOperands.size()), static_cast<int32_t>(waitDevnumOperands.size()), static_cast<int32_t>(ifCondOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(waitOperandsOperands, waitOperandsTypes, waitOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(asyncOperandOperands, asyncOperandTypes, asyncOperandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(waitDevnumOperands, waitDevnumTypes, waitDevnumOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ifCondOperands, odsBuildableType0, ifCondOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!waitOperands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << waitOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << waitOperands().getTypes();
  _odsPrinter << ")";
  }
  if (asyncOperand()) {
  _odsPrinter << ' ' << "async";
  _odsPrinter << "(";
  if (::mlir::Value value = asyncOperand())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (asyncOperand() ? ::llvm::ArrayRef<::mlir::Type>(asyncOperand().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (waitDevnum()) {
  _odsPrinter << ' ' << "wait_devnum";
  _odsPrinter << "(";
  if (::mlir::Value value = waitDevnum())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (waitDevnum() ? ::llvm::ArrayRef<::mlir::Type>(waitDevnum().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (ifCond()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = ifCond())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::WaitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("acc.yield", odsAttrs.getContext());
}

YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::operandsMutable() {
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

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenACCOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  _odsPrinter << ' ';
  _odsPrinter << operands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << operands().getTypes();
  }
}

} // namespace acc
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::acc::YieldOp)


#endif  // GET_OP_CLASSES

