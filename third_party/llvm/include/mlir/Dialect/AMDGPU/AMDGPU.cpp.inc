/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::amdgpu::RawBufferAtomicFaddOp,
::mlir::amdgpu::RawBufferLoadOp,
::mlir::amdgpu::RawBufferStoreOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace amdgpu {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isF32()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit float, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isBF16())) || ((type.isF16())) || ((type.isF32())) || ((type.isSignlessInteger(32))) || ((type.isSignlessInteger(8))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isBF16())); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 16))))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be bfloat16 type or 16-bit float or 32-bit float or 32-bit signless integer or 8-bit signless integer or vector of 32-bit float or 32-bit signless integer values of length 2/4 or vector of 16-bit float or bfloat16 type values of length 2/4/8 or vector of 8-bit signless integer values of length 2/4/8/16, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AMDGPU0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AMDGPU1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicFaddOp definitions
//===----------------------------------------------------------------------===//

RawBufferAtomicFaddOpAdaptor::RawBufferAtomicFaddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.raw_buffer_atomic_fadd", odsAttrs.getContext());
}

RawBufferAtomicFaddOpAdaptor::RawBufferAtomicFaddOpAdaptor(RawBufferAtomicFaddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RawBufferAtomicFaddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RawBufferAtomicFaddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferAtomicFaddOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange RawBufferAtomicFaddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferAtomicFaddOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value RawBufferAtomicFaddOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange RawBufferAtomicFaddOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::Value RawBufferAtomicFaddOpAdaptor::sgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr RawBufferAtomicFaddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr RawBufferAtomicFaddOpAdaptor::targetIsRDNAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, RawBufferAtomicFaddOp::targetIsRDNAAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
    return attr;
}

bool RawBufferAtomicFaddOpAdaptor::targetIsRDNA() {
  auto attr = targetIsRDNAAttr();
  return attr.getValue();
}

::mlir::BoolAttr RawBufferAtomicFaddOpAdaptor::boundsCheckAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RawBufferAtomicFaddOp::boundsCheckAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(true);
    return attr;
}

bool RawBufferAtomicFaddOpAdaptor::boundsCheck() {
  auto attr = boundsCheckAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(true).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferAtomicFaddOpAdaptor::indexOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RawBufferAtomicFaddOp::indexOffsetAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint32_t> RawBufferAtomicFaddOpAdaptor::indexOffset() {
  auto attr = indexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult RawBufferAtomicFaddOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RawBufferAtomicFaddOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RawBufferAtomicFaddOp::boundsCheckAttrName(*odsOpName)) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == RawBufferAtomicFaddOp::indexOffsetAttrName(*odsOpName)) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_targetIsRDNA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""requires attribute 'targetIsRDNA'");
    if (namedAttrIt->getName() == RawBufferAtomicFaddOp::targetIsRDNAAttrName(*odsOpName)) {
      tblgen_targetIsRDNA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_targetIsRDNA && !((tblgen_targetIsRDNA.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""attribute 'targetIsRDNA' failed to satisfy constraint: bool attribute");

  if (tblgen_boundsCheck && !((tblgen_boundsCheck.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""attribute 'boundsCheck' failed to satisfy constraint: bool attribute");

  if (tblgen_indexOffset && !(((tblgen_indexOffset.isa<::mlir::IntegerAttr>())) && ((tblgen_indexOffset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""attribute 'indexOffset' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RawBufferAtomicFaddOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range RawBufferAtomicFaddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferAtomicFaddOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value RawBufferAtomicFaddOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range RawBufferAtomicFaddOp::indices() {
  return getODSOperands(2);
}

::mlir::Value RawBufferAtomicFaddOp::sgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::sgprOffsetMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RawBufferAtomicFaddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RawBufferAtomicFaddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr RawBufferAtomicFaddOp::targetIsRDNAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, targetIsRDNAAttrName()).cast<::mlir::BoolAttr>();
}

bool RawBufferAtomicFaddOp::targetIsRDNA() {
  auto attr = targetIsRDNAAttr();
  return attr.getValue();
}

::mlir::BoolAttr RawBufferAtomicFaddOp::boundsCheckAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, boundsCheckAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool RawBufferAtomicFaddOp::boundsCheck() {
  auto attr = boundsCheckAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(true).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferAtomicFaddOp::indexOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, indexOffsetAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> RawBufferAtomicFaddOp::indexOffset() {
  auto attr = indexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void RawBufferAtomicFaddOp::targetIsRDNAAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(targetIsRDNAAttrName(), attr);
}

void RawBufferAtomicFaddOp::boundsCheckAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(boundsCheckAttrName(), attr);
}

void RawBufferAtomicFaddOp::indexOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(indexOffsetAttrName(), attr);
}

::mlir::Attribute RawBufferAtomicFaddOp::removeIndexOffsetAttr() {
  return (*this)->removeAttr(indexOffsetAttrName());
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::BoolAttr targetIsRDNA, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), targetIsRDNA);
  if (boundsCheck) {
  odsState.addAttribute(boundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::BoolAttr targetIsRDNA, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), targetIsRDNA);
  if (boundsCheck) {
  odsState.addAttribute(boundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, bool targetIsRDNA, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), odsBuilder.getBoolAttr(targetIsRDNA));
  odsState.addAttribute(boundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, bool targetIsRDNA, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), odsBuilder.getBoolAttr(targetIsRDNA));
  odsState.addAttribute(boundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RawBufferAtomicFaddOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == boundsCheckAttrName()) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == indexOffsetAttrName()) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_targetIsRDNA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'targetIsRDNA'");
    if (namedAttrIt->getName() == targetIsRDNAAttrName()) {
      tblgen_targetIsRDNA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_targetIsRDNA, "targetIsRDNA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_boundsCheck, "boundsCheck")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU1(*this, tblgen_indexOffset, "indexOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({getElementTypeOrSelf((*this->getODSOperands(0).begin())), getElementTypeOrSelf((*this->getODSOperands(1).begin()))})))))
    return emitOpError("failed to verify that all of {value, memref} have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult RawBufferAtomicFaddOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RawBufferAtomicFaddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sgprOffsetOperands;
  ::llvm::SMLoc sgprOffsetOperandsLoc;
  (void)sgprOffsetOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> indicesTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("sgprOffset"))) {

  {
    sgprOffsetOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sgprOffsetOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::FloatType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(indicesTypes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, 1, static_cast<int32_t>(indicesOperands.size()), static_cast<int32_t>(sgprOffsetOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, indicesTypes, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sgprOffsetOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RawBufferAtomicFaddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ';
  _odsPrinter << value();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << memref();
  _odsPrinter << "[";
  _odsPrinter << indices();
  _odsPrinter << "]";
  if (sgprOffset()) {
  _odsPrinter << ' ' << "sgprOffset";
  _odsPrinter << ' ';
  if (::mlir::Value value = sgprOffset())
    _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = value().getType();
    if (auto validType = type.dyn_cast<::mlir::FloatType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = memref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << indices().getTypes();
}

void RawBufferAtomicFaddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicFaddOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferLoadOp definitions
//===----------------------------------------------------------------------===//

RawBufferLoadOpAdaptor::RawBufferLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.raw_buffer_load", odsAttrs.getContext());
}

RawBufferLoadOpAdaptor::RawBufferLoadOpAdaptor(RawBufferLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RawBufferLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RawBufferLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferLoadOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange RawBufferLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferLoadOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange RawBufferLoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value RawBufferLoadOpAdaptor::sgprOffset() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr RawBufferLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr RawBufferLoadOpAdaptor::targetIsRDNAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, RawBufferLoadOp::targetIsRDNAAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
    return attr;
}

bool RawBufferLoadOpAdaptor::targetIsRDNA() {
  auto attr = targetIsRDNAAttr();
  return attr.getValue();
}

::mlir::BoolAttr RawBufferLoadOpAdaptor::boundsCheckAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RawBufferLoadOp::boundsCheckAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(true);
    return attr;
}

bool RawBufferLoadOpAdaptor::boundsCheck() {
  auto attr = boundsCheckAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(true).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferLoadOpAdaptor::indexOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RawBufferLoadOp::indexOffsetAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint32_t> RawBufferLoadOpAdaptor::indexOffset() {
  auto attr = indexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult RawBufferLoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_load' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RawBufferLoadOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RawBufferLoadOp::boundsCheckAttrName(*odsOpName)) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == RawBufferLoadOp::indexOffsetAttrName(*odsOpName)) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_targetIsRDNA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_load' op ""requires attribute 'targetIsRDNA'");
    if (namedAttrIt->getName() == RawBufferLoadOp::targetIsRDNAAttrName(*odsOpName)) {
      tblgen_targetIsRDNA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'amdgpu.raw_buffer_load' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_targetIsRDNA && !((tblgen_targetIsRDNA.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_load' op ""attribute 'targetIsRDNA' failed to satisfy constraint: bool attribute");

  if (tblgen_boundsCheck && !((tblgen_boundsCheck.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_load' op ""attribute 'boundsCheck' failed to satisfy constraint: bool attribute");

  if (tblgen_indexOffset && !(((tblgen_indexOffset.isa<::mlir::IntegerAttr>())) && ((tblgen_indexOffset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.raw_buffer_load' op ""attribute 'indexOffset' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RawBufferLoadOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range RawBufferLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferLoadOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range RawBufferLoadOp::indices() {
  return getODSOperands(1);
}

::mlir::Value RawBufferLoadOp::sgprOffset() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange RawBufferLoadOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferLoadOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferLoadOp::sgprOffsetMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RawBufferLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RawBufferLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferLoadOp::value() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr RawBufferLoadOp::targetIsRDNAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, targetIsRDNAAttrName()).cast<::mlir::BoolAttr>();
}

bool RawBufferLoadOp::targetIsRDNA() {
  auto attr = targetIsRDNAAttr();
  return attr.getValue();
}

::mlir::BoolAttr RawBufferLoadOp::boundsCheckAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, boundsCheckAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool RawBufferLoadOp::boundsCheck() {
  auto attr = boundsCheckAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(true).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferLoadOp::indexOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, indexOffsetAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> RawBufferLoadOp::indexOffset() {
  auto attr = indexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void RawBufferLoadOp::targetIsRDNAAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(targetIsRDNAAttrName(), attr);
}

void RawBufferLoadOp::boundsCheckAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(boundsCheckAttrName(), attr);
}

void RawBufferLoadOp::indexOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(indexOffsetAttrName(), attr);
}

::mlir::Attribute RawBufferLoadOp::removeIndexOffsetAttr() {
  return (*this)->removeAttr(indexOffsetAttrName());
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::BoolAttr targetIsRDNA, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), targetIsRDNA);
  if (boundsCheck) {
  odsState.addAttribute(boundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  odsState.addTypes(value);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::BoolAttr targetIsRDNA, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), targetIsRDNA);
  if (boundsCheck) {
  odsState.addAttribute(boundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, bool targetIsRDNA, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), odsBuilder.getBoolAttr(targetIsRDNA));
  odsState.addAttribute(boundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  odsState.addTypes(value);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, bool targetIsRDNA, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), odsBuilder.getBoolAttr(targetIsRDNA));
  odsState.addAttribute(boundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RawBufferLoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == boundsCheckAttrName()) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == indexOffsetAttrName()) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_targetIsRDNA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'targetIsRDNA'");
    if (namedAttrIt->getName() == targetIsRDNAAttrName()) {
      tblgen_targetIsRDNA = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_targetIsRDNA, "targetIsRDNA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_boundsCheck, "boundsCheck")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU1(*this, tblgen_indexOffset, "indexOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({getElementTypeOrSelf((*this->getODSResults(0).begin())), getElementTypeOrSelf((*this->getODSOperands(0).begin()))})))))
    return emitOpError("failed to verify that all of {value, memref} have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult RawBufferLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RawBufferLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sgprOffsetOperands;
  ::llvm::SMLoc sgprOffsetOperandsLoc;
  (void)sgprOffsetOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> indicesTypes;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("sgprOffset"))) {

  {
    sgprOffsetOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sgprOffsetOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(indicesTypes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(indicesOperands.size()), static_cast<int32_t>(sgprOffsetOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(valueTypes);
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, indicesTypes, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sgprOffsetOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RawBufferLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ';
  _odsPrinter << memref();
  _odsPrinter << "[";
  _odsPrinter << indices();
  _odsPrinter << "]";
  if (sgprOffset()) {
  _odsPrinter << ' ' << "sgprOffset";
  _odsPrinter << ' ';
  if (::mlir::Value value = sgprOffset())
    _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = memref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << indices().getTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = value().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void RawBufferLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferLoadOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferStoreOp definitions
//===----------------------------------------------------------------------===//

RawBufferStoreOpAdaptor::RawBufferStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.raw_buffer_store", odsAttrs.getContext());
}

RawBufferStoreOpAdaptor::RawBufferStoreOpAdaptor(RawBufferStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RawBufferStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RawBufferStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferStoreOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange RawBufferStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferStoreOpAdaptor::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value RawBufferStoreOpAdaptor::memref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange RawBufferStoreOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::Value RawBufferStoreOpAdaptor::sgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr RawBufferStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr RawBufferStoreOpAdaptor::targetIsRDNAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, RawBufferStoreOp::targetIsRDNAAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
    return attr;
}

bool RawBufferStoreOpAdaptor::targetIsRDNA() {
  auto attr = targetIsRDNAAttr();
  return attr.getValue();
}

::mlir::BoolAttr RawBufferStoreOpAdaptor::boundsCheckAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RawBufferStoreOp::boundsCheckAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(true);
    return attr;
}

bool RawBufferStoreOpAdaptor::boundsCheck() {
  auto attr = boundsCheckAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(true).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferStoreOpAdaptor::indexOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, RawBufferStoreOp::indexOffsetAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint32_t> RawBufferStoreOpAdaptor::indexOffset() {
  auto attr = indexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult RawBufferStoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_store' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RawBufferStoreOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RawBufferStoreOp::boundsCheckAttrName(*odsOpName)) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == RawBufferStoreOp::indexOffsetAttrName(*odsOpName)) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_targetIsRDNA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_store' op ""requires attribute 'targetIsRDNA'");
    if (namedAttrIt->getName() == RawBufferStoreOp::targetIsRDNAAttrName(*odsOpName)) {
      tblgen_targetIsRDNA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'amdgpu.raw_buffer_store' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_targetIsRDNA && !((tblgen_targetIsRDNA.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_store' op ""attribute 'targetIsRDNA' failed to satisfy constraint: bool attribute");

  if (tblgen_boundsCheck && !((tblgen_boundsCheck.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_store' op ""attribute 'boundsCheck' failed to satisfy constraint: bool attribute");

  if (tblgen_indexOffset && !(((tblgen_indexOffset.isa<::mlir::IntegerAttr>())) && ((tblgen_indexOffset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.raw_buffer_store' op ""attribute 'indexOffset' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RawBufferStoreOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range RawBufferStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferStoreOp::value() {
  return *getODSOperands(0).begin();
}

::mlir::Value RawBufferStoreOp::memref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range RawBufferStoreOp::indices() {
  return getODSOperands(2);
}

::mlir::Value RawBufferStoreOp::sgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange RawBufferStoreOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferStoreOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferStoreOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferStoreOp::sgprOffsetMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RawBufferStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RawBufferStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr RawBufferStoreOp::targetIsRDNAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, targetIsRDNAAttrName()).cast<::mlir::BoolAttr>();
}

bool RawBufferStoreOp::targetIsRDNA() {
  auto attr = targetIsRDNAAttr();
  return attr.getValue();
}

::mlir::BoolAttr RawBufferStoreOp::boundsCheckAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, boundsCheckAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool RawBufferStoreOp::boundsCheck() {
  auto attr = boundsCheckAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(true).getValue();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferStoreOp::indexOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, indexOffsetAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> RawBufferStoreOp::indexOffset() {
  auto attr = indexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void RawBufferStoreOp::targetIsRDNAAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(targetIsRDNAAttrName(), attr);
}

void RawBufferStoreOp::boundsCheckAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(boundsCheckAttrName(), attr);
}

void RawBufferStoreOp::indexOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(indexOffsetAttrName(), attr);
}

::mlir::Attribute RawBufferStoreOp::removeIndexOffsetAttr() {
  return (*this)->removeAttr(indexOffsetAttrName());
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::BoolAttr targetIsRDNA, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), targetIsRDNA);
  if (boundsCheck) {
  odsState.addAttribute(boundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::BoolAttr targetIsRDNA, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), targetIsRDNA);
  if (boundsCheck) {
  odsState.addAttribute(boundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, bool targetIsRDNA, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), odsBuilder.getBoolAttr(targetIsRDNA));
  odsState.addAttribute(boundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, bool targetIsRDNA, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(targetIsRDNAAttrName(odsState.name), odsBuilder.getBoolAttr(targetIsRDNA));
  odsState.addAttribute(boundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(indexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RawBufferStoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == boundsCheckAttrName()) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == indexOffsetAttrName()) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_targetIsRDNA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'targetIsRDNA'");
    if (namedAttrIt->getName() == targetIsRDNAAttrName()) {
      tblgen_targetIsRDNA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_targetIsRDNA, "targetIsRDNA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_boundsCheck, "boundsCheck")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU1(*this, tblgen_indexOffset, "indexOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({getElementTypeOrSelf((*this->getODSOperands(0).begin())), getElementTypeOrSelf((*this->getODSOperands(1).begin()))})))))
    return emitOpError("failed to verify that all of {value, memref} have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult RawBufferStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RawBufferStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sgprOffsetOperands;
  ::llvm::SMLoc sgprOffsetOperandsLoc;
  (void)sgprOffsetOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> indicesTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("sgprOffset"))) {

  {
    sgprOffsetOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sgprOffsetOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(indicesTypes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, 1, static_cast<int32_t>(indicesOperands.size()), static_cast<int32_t>(sgprOffsetOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, indicesTypes, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sgprOffsetOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RawBufferStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ';
  _odsPrinter << value();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << memref();
  _odsPrinter << "[";
  _odsPrinter << indices();
  _odsPrinter << "]";
  if (sgprOffset()) {
  _odsPrinter << ' ' << "sgprOffset";
  _odsPrinter << ' ';
  if (::mlir::Value value = sgprOffset())
    _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = value().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = memref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << indices().getTypes();
}

void RawBufferStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferStoreOp)


#endif  // GET_OP_CLASSES

