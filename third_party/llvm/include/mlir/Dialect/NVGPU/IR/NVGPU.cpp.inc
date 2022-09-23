/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::nvgpu::DeviceAsyncCopyOp,
::mlir::nvgpu::DeviceAsyncCreateGroupOp,
::mlir::nvgpu::DeviceAsyncWaitOp,
::mlir::nvgpu::LdMatrixOp,
::mlir::nvgpu::MmaSyncOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace nvgpu {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVGPU0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVGPU1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVGPU2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::nvgpu::DeviceAsyncTokenType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be device async token type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVGPU3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVGPU0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: index attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVGPU1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVGPU2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVGPU3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVGPU4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}
} // namespace nvgpu
} // namespace mlir
namespace mlir {
namespace nvgpu {

//===----------------------------------------------------------------------===//
// ::mlir::nvgpu::DeviceAsyncCopyOp definitions
//===----------------------------------------------------------------------===//

DeviceAsyncCopyOpAdaptor::DeviceAsyncCopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvgpu.device_async_copy", odsAttrs.getContext());
}

DeviceAsyncCopyOpAdaptor::DeviceAsyncCopyOpAdaptor(DeviceAsyncCopyOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DeviceAsyncCopyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DeviceAsyncCopyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, DeviceAsyncCopyOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange DeviceAsyncCopyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeviceAsyncCopyOpAdaptor::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange DeviceAsyncCopyOpAdaptor::getDstIndices() {
  return getODSOperands(1);
}

::mlir::Value DeviceAsyncCopyOpAdaptor::getSrc() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange DeviceAsyncCopyOpAdaptor::getSrcIndices() {
  return getODSOperands(3);
}

::mlir::Value DeviceAsyncCopyOpAdaptor::getSrcElements() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr DeviceAsyncCopyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr DeviceAsyncCopyOpAdaptor::getDstElementsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, DeviceAsyncCopyOp::getDstElementsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

::llvm::APInt DeviceAsyncCopyOpAdaptor::getDstElements() {
  auto attr = getDstElementsAttr();
  return attr.getValue();
}

::mlir::UnitAttr DeviceAsyncCopyOpAdaptor::getBypassL1Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, DeviceAsyncCopyOp::getBypassL1AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::llvm::Optional<bool> DeviceAsyncCopyOpAdaptor::getBypassL1() {
  auto attr = getBypassL1Attr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

::mlir::LogicalResult DeviceAsyncCopyOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dstElements;
  ::mlir::Attribute tblgen_bypassL1;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvgpu.device_async_copy' op ""requires attribute 'dstElements'");
    if (namedAttrIt->getName() == DeviceAsyncCopyOp::getDstElementsAttrName(*odsOpName)) {
      tblgen_dstElements = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == DeviceAsyncCopyOp::getBypassL1AttrName(*odsOpName)) {
      tblgen_bypassL1 = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvgpu.device_async_copy' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == DeviceAsyncCopyOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'nvgpu.device_async_copy' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_dstElements && !(((tblgen_dstElements.isa<::mlir::IntegerAttr>())) && ((tblgen_dstElements.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'nvgpu.device_async_copy' op ""attribute 'dstElements' failed to satisfy constraint: index attribute");

  if (tblgen_bypassL1 && !((tblgen_bypassL1.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'nvgpu.device_async_copy' op ""attribute 'bypassL1' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DeviceAsyncCopyOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range DeviceAsyncCopyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> DeviceAsyncCopyOp::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range DeviceAsyncCopyOp::getDstIndices() {
  return getODSOperands(1);
}

::mlir::TypedValue<::mlir::MemRefType> DeviceAsyncCopyOp::getSrc() {
  return *getODSOperands(2).begin();
}

::mlir::Operation::operand_range DeviceAsyncCopyOp::getSrcIndices() {
  return getODSOperands(3);
}

::mlir::TypedValue<::mlir::IndexType> DeviceAsyncCopyOp::getSrcElements() {
  auto operands = getODSOperands(4);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange DeviceAsyncCopyOp::getDstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DeviceAsyncCopyOp::getDstIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DeviceAsyncCopyOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DeviceAsyncCopyOp::getSrcIndicesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange DeviceAsyncCopyOp::getSrcElementsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> DeviceAsyncCopyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeviceAsyncCopyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeviceAsyncCopyOp::getAsyncToken() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr DeviceAsyncCopyOp::getDstElementsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDstElementsAttrName()).cast<::mlir::IntegerAttr>();
}

::llvm::APInt DeviceAsyncCopyOp::getDstElements() {
  auto attr = getDstElementsAttr();
  return attr.getValue();
}

::mlir::UnitAttr DeviceAsyncCopyOp::getBypassL1Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBypassL1AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

::llvm::Optional<bool> DeviceAsyncCopyOp::getBypassL1() {
  auto attr = getBypassL1Attr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

void DeviceAsyncCopyOp::setDstElementsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDstElementsAttrName(), attr);
}

void DeviceAsyncCopyOp::setBypassL1Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getBypassL1AttrName(), attr);
}

::mlir::Attribute DeviceAsyncCopyOp::removeBypassL1Attr() {
  return (*this)->removeAttr(getBypassL1AttrName());
}

void DeviceAsyncCopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type asyncToken, ::mlir::Value dst, ::mlir::ValueRange dstIndices, ::mlir::Value src, ::mlir::ValueRange srcIndices, ::mlir::IntegerAttr dstElements, /*optional*/::mlir::Value srcElements, /*optional*/::mlir::UnitAttr bypassL1) {
  odsState.addOperands(dst);
  odsState.addOperands(dstIndices);
  odsState.addOperands(src);
  odsState.addOperands(srcIndices);
  if (srcElements)
    odsState.addOperands(srcElements);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(dstIndices.size()), 1, static_cast<int32_t>(srcIndices.size()), (srcElements ? 1 : 0)}));
  odsState.addAttribute(getDstElementsAttrName(odsState.name), dstElements);
  if (bypassL1) {
  odsState.addAttribute(getBypassL1AttrName(odsState.name), bypassL1);
  }
  odsState.addTypes(asyncToken);
}

void DeviceAsyncCopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::ValueRange dstIndices, ::mlir::Value src, ::mlir::ValueRange srcIndices, ::mlir::IntegerAttr dstElements, /*optional*/::mlir::Value srcElements, /*optional*/::mlir::UnitAttr bypassL1) {
  odsState.addOperands(dst);
  odsState.addOperands(dstIndices);
  odsState.addOperands(src);
  odsState.addOperands(srcIndices);
  if (srcElements)
    odsState.addOperands(srcElements);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(dstIndices.size()), 1, static_cast<int32_t>(srcIndices.size()), (srcElements ? 1 : 0)}));
  odsState.addAttribute(getDstElementsAttrName(odsState.name), dstElements);
  if (bypassL1) {
  odsState.addAttribute(getBypassL1AttrName(odsState.name), bypassL1);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeviceAsyncCopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type asyncToken, ::mlir::Value dst, ::mlir::ValueRange dstIndices, ::mlir::Value src, ::mlir::ValueRange srcIndices, ::llvm::APInt dstElements, /*optional*/::mlir::Value srcElements, /*optional*/::mlir::UnitAttr bypassL1) {
  odsState.addOperands(dst);
  odsState.addOperands(dstIndices);
  odsState.addOperands(src);
  odsState.addOperands(srcIndices);
  if (srcElements)
    odsState.addOperands(srcElements);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(dstIndices.size()), 1, static_cast<int32_t>(srcIndices.size()), (srcElements ? 1 : 0)}));
  odsState.addAttribute(getDstElementsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dstElements));
  if (bypassL1) {
  odsState.addAttribute(getBypassL1AttrName(odsState.name), bypassL1);
  }
  odsState.addTypes(asyncToken);
}

void DeviceAsyncCopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::ValueRange dstIndices, ::mlir::Value src, ::mlir::ValueRange srcIndices, ::llvm::APInt dstElements, /*optional*/::mlir::Value srcElements, /*optional*/::mlir::UnitAttr bypassL1) {
  odsState.addOperands(dst);
  odsState.addOperands(dstIndices);
  odsState.addOperands(src);
  odsState.addOperands(srcIndices);
  if (srcElements)
    odsState.addOperands(srcElements);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(dstIndices.size()), 1, static_cast<int32_t>(srcIndices.size()), (srcElements ? 1 : 0)}));
  odsState.addAttribute(getDstElementsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dstElements));
  if (bypassL1) {
  odsState.addAttribute(getBypassL1AttrName(odsState.name), bypassL1);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeviceAsyncCopyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeviceAsyncCopyOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dstElements;
  ::mlir::Attribute tblgen_bypassL1;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dstElements'");
    if (namedAttrIt->getName() == getDstElementsAttrName()) {
      tblgen_dstElements = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBypassL1AttrName()) {
      tblgen_bypassL1 = namedAttrIt->getValue();
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
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU0(*this, tblgen_dstElements, "dstElements")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU1(*this, tblgen_bypassL1, "bypassL1")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    if (valueGroup4.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup4.size();
    }

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DeviceAsyncCopyOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DeviceAsyncCopyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> srcIndicesOperands;
  ::llvm::SMLoc srcIndicesOperandsLoc;
  (void)srcIndicesOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(dstRawOperands);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dstIndicesOperands;
  ::llvm::SMLoc dstIndicesOperandsLoc;
  (void)dstIndicesOperandsLoc;
  ::mlir::IntegerAttr dstElementsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> srcElementsOperands;
  ::llvm::SMLoc srcElementsOperandsLoc;
  (void)srcElementsOperandsLoc;
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  ::mlir::Type dstRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dstTypes(dstRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  srcIndicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(srcIndicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  dstIndicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dstIndicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(dstElementsAttr, parser.getBuilder().getIndexType(), "dstElements",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalComma())) {

  {
    srcElementsOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      srcElementsOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dstRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(dstIndicesOperands.size()), 1, static_cast<int32_t>(srcIndicesOperands.size()), static_cast<int32_t>(srcElementsOperands.size())}));
  ::mlir::Type odsBuildableType0 = mlir::nvgpu::DeviceAsyncTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(dstOperands, dstTypes, dstOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dstIndicesOperands, odsBuildableType1, dstIndicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcIndicesOperands, odsBuildableType1, srcIndicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcElementsOperands, odsBuildableType1, srcElementsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeviceAsyncCopyOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << "[";
  _odsPrinter << getSrcIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDst();
  _odsPrinter << "[";
  _odsPrinter << getDstIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getDstElementsAttr());
  if (getSrcElements()) {
  _odsPrinter << ",";
  _odsPrinter << ' ';
  if (::mlir::Value value = getSrcElements())
    _odsPrinter << value;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "dstElements"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getDst().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void DeviceAsyncCopyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(2))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::DeviceAsyncCopyOp)

namespace mlir {
namespace nvgpu {

//===----------------------------------------------------------------------===//
// ::mlir::nvgpu::DeviceAsyncCreateGroupOp definitions
//===----------------------------------------------------------------------===//

DeviceAsyncCreateGroupOpAdaptor::DeviceAsyncCreateGroupOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvgpu.device_async_create_group", odsAttrs.getContext());
}

DeviceAsyncCreateGroupOpAdaptor::DeviceAsyncCreateGroupOpAdaptor(DeviceAsyncCreateGroupOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DeviceAsyncCreateGroupOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DeviceAsyncCreateGroupOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange DeviceAsyncCreateGroupOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange DeviceAsyncCreateGroupOpAdaptor::getInputTokens() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr DeviceAsyncCreateGroupOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DeviceAsyncCreateGroupOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DeviceAsyncCreateGroupOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range DeviceAsyncCreateGroupOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DeviceAsyncCreateGroupOp::getInputTokens() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange DeviceAsyncCreateGroupOp::getInputTokensMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DeviceAsyncCreateGroupOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeviceAsyncCreateGroupOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeviceAsyncCreateGroupOp::getAsyncToken() {
  return *getODSResults(0).begin();
}

void DeviceAsyncCreateGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type asyncToken, ::mlir::ValueRange inputTokens) {
  odsState.addOperands(inputTokens);
  odsState.addTypes(asyncToken);
}

void DeviceAsyncCreateGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeviceAsyncCreateGroupOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DeviceAsyncCreateGroupOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DeviceAsyncCreateGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputTokensOperands;
  ::llvm::SMLoc inputTokensOperandsLoc;
  (void)inputTokensOperandsLoc;

  inputTokensOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputTokensOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::nvgpu::DeviceAsyncTokenType::get(parser.getBuilder().getContext());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputTokensOperands, odsBuildableType0, inputTokensOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeviceAsyncCreateGroupOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputTokens();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::DeviceAsyncCreateGroupOp)

namespace mlir {
namespace nvgpu {

//===----------------------------------------------------------------------===//
// ::mlir::nvgpu::DeviceAsyncWaitOp definitions
//===----------------------------------------------------------------------===//

DeviceAsyncWaitOpAdaptor::DeviceAsyncWaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvgpu.device_async_wait", odsAttrs.getContext());
}

DeviceAsyncWaitOpAdaptor::DeviceAsyncWaitOpAdaptor(DeviceAsyncWaitOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DeviceAsyncWaitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DeviceAsyncWaitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DeviceAsyncWaitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeviceAsyncWaitOpAdaptor::getAsyncDependencies() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr DeviceAsyncWaitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr DeviceAsyncWaitOpAdaptor::getNumGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, DeviceAsyncWaitOp::getNumGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint32_t> DeviceAsyncWaitOpAdaptor::getNumGroups() {
  auto attr = getNumGroupsAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult DeviceAsyncWaitOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_numGroups;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == DeviceAsyncWaitOp::getNumGroupsAttrName(*odsOpName)) {
      tblgen_numGroups = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_numGroups && !(((tblgen_numGroups.isa<::mlir::IntegerAttr>())) && ((tblgen_numGroups.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvgpu.device_async_wait' op ""attribute 'numGroups' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DeviceAsyncWaitOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DeviceAsyncWaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DeviceAsyncWaitOp::getAsyncDependencies() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DeviceAsyncWaitOp::getAsyncDependenciesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DeviceAsyncWaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeviceAsyncWaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr DeviceAsyncWaitOp::getNumGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNumGroupsAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> DeviceAsyncWaitOp::getNumGroups() {
  auto attr = getNumGroupsAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void DeviceAsyncWaitOp::setNumGroupsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumGroupsAttrName(), attr);
}

::mlir::Attribute DeviceAsyncWaitOp::removeNumGroupsAttr() {
  return (*this)->removeAttr(getNumGroupsAttrName());
}

void DeviceAsyncWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value asyncDependencies, /*optional*/::mlir::IntegerAttr numGroups) {
  odsState.addOperands(asyncDependencies);
  if (numGroups) {
  odsState.addAttribute(getNumGroupsAttrName(odsState.name), numGroups);
  }
}

void DeviceAsyncWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value asyncDependencies, /*optional*/::mlir::IntegerAttr numGroups) {
  odsState.addOperands(asyncDependencies);
  if (numGroups) {
  odsState.addAttribute(getNumGroupsAttrName(odsState.name), numGroups);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeviceAsyncWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeviceAsyncWaitOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_numGroups;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNumGroupsAttrName()) {
      tblgen_numGroups = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU2(*this, tblgen_numGroups, "numGroups")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DeviceAsyncWaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DeviceAsyncWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand asyncDependenciesRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> asyncDependenciesOperands(asyncDependenciesRawOperands);  ::llvm::SMLoc asyncDependenciesOperandsLoc;
  (void)asyncDependenciesOperandsLoc;

  asyncDependenciesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(asyncDependenciesRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = mlir::nvgpu::DeviceAsyncTokenType::get(parser.getBuilder().getContext());
  if (parser.resolveOperands(asyncDependenciesOperands, odsBuildableType0, asyncDependenciesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeviceAsyncWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAsyncDependencies();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::DeviceAsyncWaitOp)

namespace mlir {
namespace nvgpu {

//===----------------------------------------------------------------------===//
// ::mlir::nvgpu::LdMatrixOp definitions
//===----------------------------------------------------------------------===//

LdMatrixOpAdaptor::LdMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvgpu.ldmatrix", odsAttrs.getContext());
}

LdMatrixOpAdaptor::LdMatrixOpAdaptor(LdMatrixOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LdMatrixOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LdMatrixOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LdMatrixOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LdMatrixOpAdaptor::getSrcMemref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange LdMatrixOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr LdMatrixOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr LdMatrixOpAdaptor::getTransposeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LdMatrixOp::getTransposeAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool LdMatrixOpAdaptor::getTranspose() {
  auto attr = getTransposeAttr();
  return attr.getValue();
}

::mlir::IntegerAttr LdMatrixOpAdaptor::getNumTilesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LdMatrixOp::getNumTilesAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t LdMatrixOpAdaptor::getNumTiles() {
  auto attr = getNumTilesAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult LdMatrixOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_numTiles;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvgpu.ldmatrix' op ""requires attribute 'numTiles'");
    if (namedAttrIt->getName() == LdMatrixOp::getNumTilesAttrName(*odsOpName)) {
      tblgen_numTiles = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_transpose;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvgpu.ldmatrix' op ""requires attribute 'transpose'");
    if (namedAttrIt->getName() == LdMatrixOp::getTransposeAttrName(*odsOpName)) {
      tblgen_transpose = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_transpose && !((tblgen_transpose.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'nvgpu.ldmatrix' op ""attribute 'transpose' failed to satisfy constraint: bool attribute");

  if (tblgen_numTiles && !(((tblgen_numTiles.isa<::mlir::IntegerAttr>())) && ((tblgen_numTiles.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvgpu.ldmatrix' op ""attribute 'numTiles' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LdMatrixOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LdMatrixOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> LdMatrixOp::getSrcMemref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range LdMatrixOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange LdMatrixOp::getSrcMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LdMatrixOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LdMatrixOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LdMatrixOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::VectorType> LdMatrixOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr LdMatrixOp::getTransposeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTransposeAttrName()).cast<::mlir::BoolAttr>();
}

bool LdMatrixOp::getTranspose() {
  auto attr = getTransposeAttr();
  return attr.getValue();
}

::mlir::IntegerAttr LdMatrixOp::getNumTilesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNumTilesAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t LdMatrixOp::getNumTiles() {
  auto attr = getNumTilesAttr();
  return attr.getValue().getZExtValue();
}

void LdMatrixOp::setTransposeAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getTransposeAttrName(), attr);
}

void LdMatrixOp::setNumTilesAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumTilesAttrName(), attr);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::mlir::BoolAttr transpose, ::mlir::IntegerAttr numTiles) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(getTransposeAttrName(odsState.name), transpose);
  odsState.addAttribute(getNumTilesAttrName(odsState.name), numTiles);
  odsState.addTypes(res);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value srcMemref, ::mlir::ValueRange indices, ::mlir::BoolAttr transpose, ::mlir::IntegerAttr numTiles) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(getTransposeAttrName(odsState.name), transpose);
  odsState.addAttribute(getNumTilesAttrName(odsState.name), numTiles);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value srcMemref, ::mlir::ValueRange indices, bool transpose, uint32_t numTiles) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(getTransposeAttrName(odsState.name), odsBuilder.getBoolAttr(transpose));
  odsState.addAttribute(getNumTilesAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numTiles));
  odsState.addTypes(res);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value srcMemref, ::mlir::ValueRange indices, bool transpose, uint32_t numTiles) {
  odsState.addOperands(srcMemref);
  odsState.addOperands(indices);
  odsState.addAttribute(getTransposeAttrName(odsState.name), odsBuilder.getBoolAttr(transpose));
  odsState.addAttribute(getNumTilesAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), numTiles));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LdMatrixOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LdMatrixOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_numTiles;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'numTiles'");
    if (namedAttrIt->getName() == getNumTilesAttrName()) {
      tblgen_numTiles = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_transpose;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'transpose'");
    if (namedAttrIt->getName() == getTransposeAttrName()) {
      tblgen_transpose = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU3(*this, tblgen_transpose, "transpose")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU2(*this, tblgen_numTiles, "numTiles")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((((*this->getOperation()).getNumResults() > 0)) && (((*this->getOperation()).getNumOperands() > 0)) && (((*this->getOperation()).getResult(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(0).getType().isa<::mlir::ShapedType>()))) && ((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0))))))
    return emitOpError("failed to verify that srcMemref and res have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult LdMatrixOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult LdMatrixOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcMemrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcMemrefOperands(srcMemrefRawOperands);  ::llvm::SMLoc srcMemrefOperandsLoc;
  (void)srcMemrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type srcMemrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcMemrefTypes(srcMemrefRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  srcMemrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcMemrefRawOperands[0]))
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
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    srcMemrefRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resTypes);
  if (parser.resolveOperands(srcMemrefOperands, srcMemrefTypes, srcMemrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LdMatrixOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrcMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrcMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LdMatrixOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(), ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::LdMatrixOp)

namespace mlir {
namespace nvgpu {

//===----------------------------------------------------------------------===//
// ::mlir::nvgpu::MmaSyncOp definitions
//===----------------------------------------------------------------------===//

MmaSyncOpAdaptor::MmaSyncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvgpu.mma.sync", odsAttrs.getContext());
}

MmaSyncOpAdaptor::MmaSyncOpAdaptor(MmaSyncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MmaSyncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MmaSyncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MmaSyncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MmaSyncOpAdaptor::getMatrixA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MmaSyncOpAdaptor::getMatrixB() {
  return *getODSOperands(1).begin();
}

::mlir::Value MmaSyncOpAdaptor::getMatrixC() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MmaSyncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr MmaSyncOpAdaptor::getMmaShapeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MmaSyncOp::getMmaShapeAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MmaSyncOpAdaptor::getMmaShape() {
  auto attr = getMmaShapeAttr();
  return attr;
}

::mlir::UnitAttr MmaSyncOpAdaptor::getTf32EnabledAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, MmaSyncOp::getTf32EnabledAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::llvm::Optional<bool> MmaSyncOpAdaptor::getTf32Enabled() {
  auto attr = getTf32EnabledAttr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

::mlir::LogicalResult MmaSyncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_mmaShape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvgpu.mma.sync' op ""requires attribute 'mmaShape'");
    if (namedAttrIt->getName() == MmaSyncOp::getMmaShapeAttrName(*odsOpName)) {
      tblgen_mmaShape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_tf32Enabled;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MmaSyncOp::getTf32EnabledAttrName(*odsOpName)) {
      tblgen_tf32Enabled = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_mmaShape && !(((tblgen_mmaShape.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_mmaShape.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'nvgpu.mma.sync' op ""attribute 'mmaShape' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_tf32Enabled && !((tblgen_tf32Enabled.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'nvgpu.mma.sync' op ""attribute 'tf32Enabled' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MmaSyncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MmaSyncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::VectorType> MmaSyncOp::getMatrixA() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::VectorType> MmaSyncOp::getMatrixB() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::VectorType> MmaSyncOp::getMatrixC() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MmaSyncOp::getMatrixAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MmaSyncOp::getMatrixBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MmaSyncOp::getMatrixCMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MmaSyncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MmaSyncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::VectorType> MmaSyncOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr MmaSyncOp::getMmaShapeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMmaShapeAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MmaSyncOp::getMmaShape() {
  auto attr = getMmaShapeAttr();
  return attr;
}

::mlir::UnitAttr MmaSyncOp::getTf32EnabledAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTf32EnabledAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

::llvm::Optional<bool> MmaSyncOp::getTf32Enabled() {
  auto attr = getTf32EnabledAttr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

void MmaSyncOp::setMmaShapeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMmaShapeAttrName(), attr);
}

void MmaSyncOp::setTf32EnabledAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getTf32EnabledAttrName(), attr);
}

::mlir::Attribute MmaSyncOp::removeTf32EnabledAttr() {
  return (*this)->removeAttr(getTf32EnabledAttrName());
}

void MmaSyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrixA, ::mlir::Value matrixB, ::mlir::Value matrixC, ::mlir::ArrayAttr mmaShape, /*optional*/::mlir::UnitAttr tf32Enabled) {
  odsState.addOperands(matrixA);
  odsState.addOperands(matrixB);
  odsState.addOperands(matrixC);
  odsState.addAttribute(getMmaShapeAttrName(odsState.name), mmaShape);
  if (tf32Enabled) {
  odsState.addAttribute(getTf32EnabledAttrName(odsState.name), tf32Enabled);
  }
  odsState.addTypes(res);
}

void MmaSyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrixA, ::mlir::Value matrixB, ::mlir::Value matrixC, ::mlir::ArrayAttr mmaShape, /*optional*/::mlir::UnitAttr tf32Enabled) {
  odsState.addOperands(matrixA);
  odsState.addOperands(matrixB);
  odsState.addOperands(matrixC);
  odsState.addAttribute(getMmaShapeAttrName(odsState.name), mmaShape);
  if (tf32Enabled) {
  odsState.addAttribute(getTf32EnabledAttrName(odsState.name), tf32Enabled);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MmaSyncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MmaSyncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_mmaShape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'mmaShape'");
    if (namedAttrIt->getName() == getMmaShapeAttrName()) {
      tblgen_mmaShape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_tf32Enabled;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getTf32EnabledAttrName()) {
      tblgen_tf32Enabled = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU4(*this, tblgen_mmaShape, "mmaShape")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVGPU1(*this, tblgen_tf32Enabled, "tf32Enabled")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVGPU3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getOperation()).getNumOperands() > 1)) && (((*this->getOperation()).getOperand(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(1).getType().isa<::mlir::ShapedType>())) && ((::mlir::getElementTypeOrSelf((*this->getOperation()).getOperand(0)) == ::mlir::getElementTypeOrSelf((*this->getOperation()).getOperand(1))))))
    return emitOpError("failed to verify that matrixA and matrixB have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult MmaSyncOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult MmaSyncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand matrixARawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> matrixAOperands(matrixARawOperands);  ::llvm::SMLoc matrixAOperandsLoc;
  (void)matrixAOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand matrixBRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> matrixBOperands(matrixBRawOperands);  ::llvm::SMLoc matrixBOperandsLoc;
  (void)matrixBOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand matrixCRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> matrixCOperands(matrixCRawOperands);  ::llvm::SMLoc matrixCOperandsLoc;
  (void)matrixCOperandsLoc;
  ::mlir::Type matrixARawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixATypes(matrixARawTypes);
  ::mlir::Type matrixBRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixBTypes(matrixBRawTypes);
  ::mlir::Type matrixCRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixCTypes(matrixCRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  matrixAOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixARawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  matrixBOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixBRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  matrixCOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixCRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    matrixARawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    matrixBRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    matrixCRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(matrixAOperands, matrixATypes, matrixAOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matrixBOperands, matrixBTypes, matrixBOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matrixCOperands, matrixCTypes, matrixCOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MmaSyncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getMatrixA();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMatrixB();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMatrixC();
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getMatrixA().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMatrixB().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMatrixC().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MmaSyncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace nvgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::nvgpu::MmaSyncOp)


#endif  // GET_OP_CLASSES

