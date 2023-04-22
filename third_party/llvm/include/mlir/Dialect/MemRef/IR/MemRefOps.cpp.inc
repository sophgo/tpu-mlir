/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::memref::AssumeAlignmentOp,
::mlir::memref::AtomicRMWOp,
::mlir::memref::AtomicYieldOp,
::mlir::memref::CopyOp,
::mlir::memref::GenericAtomicRMWOp,
::mlir::memref::LoadOp,
::mlir::memref::AllocOp,
::mlir::memref::AllocaOp,
::mlir::memref::AllocaScopeOp,
::mlir::memref::AllocaScopeReturnOp,
::mlir::memref::CastOp,
::mlir::memref::CollapseShapeOp,
::mlir::memref::DeallocOp,
::mlir::memref::DimOp,
::mlir::memref::DmaStartOp,
::mlir::memref::DmaWaitOp,
::mlir::memref::ExpandShapeOp,
::mlir::memref::ExtractAlignedPointerAsIndexOp,
::mlir::memref::ExtractStridedMetadataOp,
::mlir::memref::GetGlobalOp,
::mlir::memref::GlobalOp,
::mlir::memref::MemorySpaceCastOp,
::mlir::memref::PrefetchOp,
::mlir::memref::RankOp,
::mlir::memref::ReallocOp,
::mlir::memref::ReinterpretCastOp,
::mlir::memref::ReshapeOp,
::mlir::memref::StoreOp,
::mlir::memref::TransposeOp,
::mlir::memref::ViewOp,
::mlir::memref::SubViewOp,
::mlir::memref::TensorStoreOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace memref {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger())) || ((type.isa<::mlir::FloatType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless integer or floating-point, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isa<::mlir::FloatType>())); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of signless integer or floating-point values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::BaseMemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked or unranked memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (( isStrided(type.cast<::mlir::MemRefType>()) )))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be strided memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::UnrankedMemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank() >= 1)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be unranked.memref of any type values or non-0-ranked.memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (( isStrided(type.cast<::mlir::MemRefType>()) ))) && ((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 0)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be strided memref of any type values of rank 0, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((type.cast<::mlir::ShapedType>().hasStaticShape())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be statically shaped memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isa<::mlir::IndexType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of signless integer or index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of 8-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MemRefOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((attr.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::arith::AtomicRMWKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getInt() >= 0)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Array of 64-bit integer array attributes";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::MemRefType>())) && ((true)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: memref type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((attr.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((attr.cast<::mlir::IntegerAttr>().getInt() <= 3)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DenseI64ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: i64 dense array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MemRefOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::AffineMapAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_MemRefOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_MemRefOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AssumeAlignmentOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AssumeAlignmentOpGenericAdaptorBase::AssumeAlignmentOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.assume_alignment", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AssumeAlignmentOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AssumeAlignmentOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AssumeAlignmentOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AssumeAlignmentOp::getAlignmentAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t AssumeAlignmentOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
AssumeAlignmentOpAdaptor::AssumeAlignmentOpAdaptor(AssumeAlignmentOp op) : AssumeAlignmentOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AssumeAlignmentOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.assume_alignment' op ""requires attribute 'alignment'");
    if (namedAttrIt->getName() == AssumeAlignmentOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'memref.assume_alignment' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AssumeAlignmentOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AssumeAlignmentOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> AssumeAlignmentOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange AssumeAlignmentOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AssumeAlignmentOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AssumeAlignmentOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AssumeAlignmentOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t AssumeAlignmentOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

void AssumeAlignmentOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AssumeAlignmentOp::setAlignment(uint32_t attrValue) {
  (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, uint32_t alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, uint32_t alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeAlignmentOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AssumeAlignmentOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'alignment'");
    if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps0(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AssumeAlignmentOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AssumeAlignmentOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::IntegerAttr alignmentAttr;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(alignmentAttr, parser.getBuilder().getIntegerType(32), "alignment",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AssumeAlignmentOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getAlignmentAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("alignment");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AssumeAlignmentOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AtomicRMWOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicRMWOpGenericAdaptorBase::AtomicRMWOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.atomic_rmw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicRMWOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr AtomicRMWOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::arith::AtomicRMWKindAttr AtomicRMWOpGenericAdaptorBase::getKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicRMWOp::getKindAttrName(*odsOpName)).cast<::mlir::arith::AtomicRMWKindAttr>();
  return attr;
}

::mlir::arith::AtomicRMWKind AtomicRMWOpGenericAdaptorBase::getKind() {
  auto attr = getKindAttr();
  return attr.getValue();
}

} // namespace detail
AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(AtomicRMWOp op) : AtomicRMWOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kind;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.atomic_rmw' op ""requires attribute 'kind'");
    if (namedAttrIt->getName() == AtomicRMWOp::getKindAttrName(*odsOpName)) {
      tblgen_kind = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_kind && !((tblgen_kind.isa<::mlir::arith::AtomicRMWKindAttr>())))
    return emitError(loc, "'memref.atomic_rmw' op ""attribute 'kind' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range AtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> AtomicRMWOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range AtomicRMWOp::getIndices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange AtomicRMWOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicRMWOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicRMWOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::arith::AtomicRMWKindAttr AtomicRMWOp::getKindAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getKindAttrName()).cast<::mlir::arith::AtomicRMWKindAttr>();
}

::mlir::arith::AtomicRMWKind AtomicRMWOp::getKind() {
  auto attr = getKindAttr();
  return attr.getValue();
}

void AtomicRMWOp::setKindAttr(::mlir::arith::AtomicRMWKindAttr attr) {
  (*this)->setAttr(getKindAttrName(), attr);
}

void AtomicRMWOp::setKind(::mlir::arith::AtomicRMWKind attrValue) {
  (*this)->setAttr(getKindAttrName(), ::mlir::arith::AtomicRMWKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::arith::AtomicRMWKindAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addTypes(result);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::arith::AtomicRMWKindAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtomicRMWOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::arith::AtomicRMWKindAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::arith::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::arith::AtomicRMWKindAttr::get(odsBuilder.getContext(), kind));
  odsState.addTypes(result);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::arith::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::arith::AtomicRMWKindAttr::get(odsBuilder.getContext(), kind));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AtomicRMWOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::arith::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::arith::AtomicRMWKindAttr::get(odsBuilder.getContext(), kind));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AtomicRMWOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AtomicRMWOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kind;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'kind'");
    if (namedAttrIt->getName() == getKindAttrName()) {
      tblgen_kind = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps1(*this, tblgen_kind, "kind")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {value, result} have same type");
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<MemRefType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that value type matches element type of memref");
  return ::mlir::success();
}

::mlir::LogicalResult AtomicRMWOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult AtomicRMWOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AtomicRMWOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::arith::AtomicRMWKindAttr kindAttr;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"addf","addi","assign","maxf","maxs","maxu","minf","mins","minu","mulf","muli","ori","andi"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "kind", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {
        return parser.emitError(loc, "expected string or keyword containing one of the following enum values for attribute 'kind' [addf, addi, assign, maxf, maxs, maxu, minf, mins, minu, mulf, muli, ori, andi]");
      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::arith::symbolizeAtomicRMWKind(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "kind attribute specification: \"" << attrStr << '"';;

      kindAttr = ::mlir::arith::AtomicRMWKindAttr::get(parser.getBuilder().getContext(), *attrOptional);
      result.addAttribute("kind", kindAttr);
    }
  }

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
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
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicRMWOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';

  {
    auto caseValue = getKind();
    auto caseValueStr = stringifyAtomicRMWKind(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("kind");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AtomicRMWOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AtomicYieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AtomicYieldOpGenericAdaptorBase::AtomicYieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.atomic_yield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AtomicYieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AtomicYieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AtomicYieldOpAdaptor::AtomicYieldOpAdaptor(AtomicYieldOp op) : AtomicYieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AtomicYieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicYieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicYieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicYieldOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange AtomicYieldOp::getResultMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicYieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicYieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AtomicYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value result) {
  odsState.addOperands(result);
}

void AtomicYieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value result) {
  odsState.addOperands(result);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicYieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicYieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicYieldOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicYieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand resultRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> resultOperands(resultRawOperands);  ::llvm::SMLoc resultOperandsLoc;
  (void)resultOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  resultOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(resultRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicYieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getResult();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AtomicYieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AtomicYieldOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CopyOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CopyOpGenericAdaptorBase::CopyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.copy", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CopyOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CopyOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CopyOpAdaptor::CopyOpAdaptor(CopyOp op) : CopyOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CopyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CopyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> CopyOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::BaseMemRefType> CopyOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange CopyOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CopyOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CopyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value target) {
  odsState.addOperands(source);
  odsState.addOperands(target);
}

void CopyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value target) {
  odsState.addOperands(source);
  odsState.addOperands(target);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CopyOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CopyOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CopyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type targetRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> targetTypes(targetRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    targetRawTypes[0] = type;
  }
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(targetOperands, targetTypes, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CopyOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getTarget().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CopyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::CopyOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GenericAtomicRMWOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GenericAtomicRMWOpGenericAdaptorBase::GenericAtomicRMWOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.generic_atomic_rmw", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GenericAtomicRMWOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr GenericAtomicRMWOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &GenericAtomicRMWOpGenericAdaptorBase::getAtomicBody() {
  return *odsRegions[0];
}

::mlir::RegionRange GenericAtomicRMWOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
GenericAtomicRMWOpAdaptor::GenericAtomicRMWOpAdaptor(GenericAtomicRMWOp op) : GenericAtomicRMWOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GenericAtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GenericAtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GenericAtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> GenericAtomicRMWOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range GenericAtomicRMWOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GenericAtomicRMWOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GenericAtomicRMWOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GenericAtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenericAtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GenericAtomicRMWOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Region &GenericAtomicRMWOp::getAtomicBody() {
  return (*this)->getRegion(0);
}

::mlir::LogicalResult GenericAtomicRMWOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType().cast<MemRefType>().getElementType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches element type of memref");
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_MemRefOps0(*this, region, "atomic_body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GenericAtomicRMWOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult GenericAtomicRMWOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType().cast<MemRefType>().getElementType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::GenericAtomicRMWOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::LoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoadOpGenericAdaptorBase::LoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.load", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr LoadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr LoadOpGenericAdaptorBase::getNontemporalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getNontemporalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

bool LoadOpGenericAdaptorBase::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

} // namespace detail
LoadOpAdaptor::LoadOpAdaptor(LoadOp op) : LoadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_nontemporal;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoadOp::getNontemporalAttrName(*odsOpName)) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_nontemporal && !((tblgen_nontemporal.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'memref.load' op ""attribute 'nontemporal' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> LoadOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range LoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange LoadOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::BoolAttr LoadOp::getNontemporalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNontemporalAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool LoadOp::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void LoadOp::setNontemporalAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getNontemporalAttrName(), attr);
}

void LoadOp::setNontemporal(::std::optional<bool> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNontemporalAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(*attrValue));
    (*this)->removeAttr(getNontemporalAttrName());
}

::mlir::Attribute LoadOp::removeNontemporalAttr() {
  return (*this)->removeAttr(getNontemporalAttrName());
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  odsState.addTypes(result);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LoadOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getBoolAttr(nontemporal));
  odsState.addTypes(result);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getBoolAttr(nontemporal));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LoadOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getBoolAttr(nontemporal));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(LoadOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult LoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_nontemporal;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNontemporalAttrName()) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps2(*this, tblgen_nontemporal, "nontemporal")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType().cast<MemRefType>().getElementType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches element type of 'memref'");
  return ::mlir::success();
}

::mlir::LogicalResult LoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult LoadOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType().cast<MemRefType>().getElementType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

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
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(memrefTypes[0].cast<MemRefType>().getElementType());
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNontemporalAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("nontemporal");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::LoadOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AllocOpGenericAdaptorBase::AllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.alloc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AllocOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr AllocOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AllocOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AllocOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> AllocOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
AllocOpAdaptor::AllocOpAdaptor(AllocOp op) : AllocOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AllocOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.alloc' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == AllocOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AllocOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'memref.alloc' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_alignment && !((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'memref.alloc' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AllocOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range AllocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AllocOp::getDynamicSizes() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range AllocOp::getSymbolOperands() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AllocOp::getDynamicSizesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange AllocOp::getSymbolOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> AllocOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> AllocOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr AllocOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> AllocOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void AllocOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AllocOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute AllocOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment) {
      return build(odsBuilder, odsState, memrefType, {}, alignment);
    
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, IntegerAttr alignment) {
      return build(odsBuilder, odsState, memrefType, dynamicSizes, {}, alignment);
    
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, ValueRange symbolOperands, IntegerAttr alignment) {
      odsState.types.push_back(memrefType);
      odsState.addOperands(dynamicSizes);
      odsState.addOperands(symbolOperands);
      odsState.addAttribute(getOperandSegmentSizeAttr(),
          odsBuilder.getDenseI32ArrayAttr({
              static_cast<int32_t>(dynamicSizes.size()),
              static_cast<int32_t>(symbolOperands.size())}));
      if (alignment)
        odsState.addAttribute(getAlignmentAttrStrName(), alignment);
    
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(dynamicSizes);
  odsState.addOperands(symbolOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(dynamicSizes.size()), static_cast<int32_t>(symbolOperands.size())}));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(memref);
}

void AllocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(dynamicSizes);
  odsState.addOperands(symbolOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(dynamicSizes.size()), static_cast<int32_t>(symbolOperands.size())}));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps3(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AllocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dynamicSizesOperands;
  ::llvm::SMLoc dynamicSizesOperandsLoc;
  (void)dynamicSizesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> symbolOperandsOperands;
  ::llvm::SMLoc symbolOperandsOperandsLoc;
  (void)symbolOperandsOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  dynamicSizesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dynamicSizesOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  symbolOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(symbolOperandsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(dynamicSizesOperands.size()), static_cast<int32_t>(symbolOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(memrefTypes);
  if (parser.resolveOperands(dynamicSizesOperands, odsBuildableType0, dynamicSizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(symbolOperandsOperands, odsBuildableType0, symbolOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getDynamicSizes();
  _odsPrinter << ")";
  if (!getSymbolOperands().empty()) {
    _odsPrinter << "[";
    _odsPrinter << getSymbolOperands();
    _odsPrinter << "]";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AllocOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AllocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AllocaOpGenericAdaptorBase::AllocaOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.alloca", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AllocaOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocaOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr AllocaOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AllocaOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AllocaOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> AllocaOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
AllocaOpAdaptor::AllocaOpAdaptor(AllocaOp op) : AllocaOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AllocaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.alloca' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == AllocaOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AllocaOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'memref.alloca' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_alignment && !((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'memref.alloca' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AllocaOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range AllocaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AllocaOp::getDynamicSizes() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range AllocaOp::getSymbolOperands() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange AllocaOp::getDynamicSizesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange AllocaOp::getSymbolOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> AllocaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> AllocaOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr AllocaOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> AllocaOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void AllocaOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AllocaOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute AllocaOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment) {
      return build(odsBuilder, odsState, memrefType, {}, alignment);
    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, IntegerAttr alignment) {
      return build(odsBuilder, odsState, memrefType, dynamicSizes, {}, alignment);
    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, ValueRange symbolOperands, IntegerAttr alignment) {
      odsState.types.push_back(memrefType);
      odsState.addOperands(dynamicSizes);
      odsState.addOperands(symbolOperands);
      odsState.addAttribute(getOperandSegmentSizeAttr(),
          odsBuilder.getDenseI32ArrayAttr({
              static_cast<int32_t>(dynamicSizes.size()),
              static_cast<int32_t>(symbolOperands.size())}));
      if (alignment)
        odsState.addAttribute(getAlignmentAttrStrName(), alignment);
    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(dynamicSizes);
  odsState.addOperands(symbolOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(dynamicSizes.size()), static_cast<int32_t>(symbolOperands.size())}));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(memref);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(dynamicSizes);
  odsState.addOperands(symbolOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(dynamicSizes.size()), static_cast<int32_t>(symbolOperands.size())}));
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps3(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocaOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AllocaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dynamicSizesOperands;
  ::llvm::SMLoc dynamicSizesOperandsLoc;
  (void)dynamicSizesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> symbolOperandsOperands;
  ::llvm::SMLoc symbolOperandsOperandsLoc;
  (void)symbolOperandsOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  dynamicSizesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(dynamicSizesOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  symbolOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(symbolOperandsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(dynamicSizesOperands.size()), static_cast<int32_t>(symbolOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(memrefTypes);
  if (parser.resolveOperands(dynamicSizesOperands, odsBuildableType0, dynamicSizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(symbolOperandsOperands, odsBuildableType0, symbolOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getDynamicSizes();
  _odsPrinter << ")";
  if (!getSymbolOperands().empty()) {
    _odsPrinter << "[";
    _odsPrinter << getSymbolOperands();
    _odsPrinter << "]";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AllocaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaScopeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AllocaScopeOpGenericAdaptorBase::AllocaScopeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.alloca_scope", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AllocaScopeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AllocaScopeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &AllocaScopeOpGenericAdaptorBase::getBodyRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange AllocaScopeOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
AllocaScopeOpAdaptor::AllocaScopeOpAdaptor(AllocaScopeOp op) : AllocaScopeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AllocaScopeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AllocaScopeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AllocaScopeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AllocaScopeOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range AllocaScopeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range AllocaScopeOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &AllocaScopeOp::getBodyRegion() {
  return (*this)->getRegion(0);
}

void AllocaScopeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results) {
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void AllocaScopeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocaScopeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_MemRefOps1(*this, region, "bodyRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocaScopeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaScopeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaScopeReturnOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AllocaScopeReturnOpGenericAdaptorBase::AllocaScopeReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.alloca_scope.return", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AllocaScopeReturnOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr AllocaScopeReturnOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AllocaScopeReturnOpAdaptor::AllocaScopeReturnOpAdaptor(AllocaScopeReturnOp op) : AllocaScopeReturnOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult AllocaScopeReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AllocaScopeReturnOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range AllocaScopeReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range AllocaScopeReturnOp::getResults() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange AllocaScopeReturnOp::getResultsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AllocaScopeReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocaScopeReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AllocaScopeReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 /*nothing to do */ 
}

void AllocaScopeReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
}

void AllocaScopeReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocaScopeReturnOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocaScopeReturnOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AllocaScopeReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> resultsOperands;
  ::llvm::SMLoc resultsOperandsLoc;
  (void)resultsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  resultsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(resultsOperands))
    return ::mlir::failure();
  if (!resultsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(resultsOperands, resultsTypes, resultsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AllocaScopeReturnOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getResults().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getResults();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
}

void AllocaScopeReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaScopeReturnOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CastOpGenericAdaptorBase::CastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.cast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CastOpAdaptor::CastOpAdaptor(CastOp op) : CastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::TypedValue<::mlir::BaseMemRefType> CastOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
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

::mlir::TypedValue<::mlir::BaseMemRefType> CastOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSResults(0).begin());
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "result", index++)))
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
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
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
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::CastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CollapseShapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CollapseShapeOpGenericAdaptorBase::CollapseShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.collapse_shape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CollapseShapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CollapseShapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CollapseShapeOpGenericAdaptorBase::getReassociationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CollapseShapeOp::getReassociationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CollapseShapeOpGenericAdaptorBase::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

} // namespace detail
CollapseShapeOpAdaptor::CollapseShapeOpAdaptor(CollapseShapeOp op) : CollapseShapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CollapseShapeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.collapse_shape' op ""requires attribute 'reassociation'");
    if (namedAttrIt->getName() == CollapseShapeOp::getReassociationAttrName(*odsOpName)) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_reassociation && !(((tblgen_reassociation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reassociation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); }))))
    return emitError(loc, "'memref.collapse_shape' op ""attribute 'reassociation' failed to satisfy constraint: Array of 64-bit integer array attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CollapseShapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CollapseShapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> CollapseShapeOp::getSrc() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CollapseShapeOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CollapseShapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CollapseShapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> CollapseShapeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::ArrayAttr CollapseShapeOp::getReassociationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReassociationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CollapseShapeOp::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

void CollapseShapeOp::setReassociationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReassociationAttrName(), attr);
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs) {
      auto reassociationMaps =
          convertReassociationMapsToIndices(odsBuilder, reassociation);
      build(odsBuilder, odsState, src, reassociationMaps, attrs);
    
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState, resultType, src, attrs);
      odsState.addAttribute("reassociation",
                          getReassociationIndicesAttribute(odsBuilder, reassociation));
    
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs) {
      auto reassociationMaps =
          convertReassociationMapsToIndices(odsBuilder, reassociation);
      build(odsBuilder, odsState, resultType, src, reassociationMaps, attrs);
    
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  odsState.addTypes(result);
}

void CollapseShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CollapseShapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CollapseShapeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'reassociation'");
    if (namedAttrIt->getName() == getReassociationAttrName()) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps4(*this, tblgen_reassociation, "reassociation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CollapseShapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CollapseShapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::ArrayAttr reassociationAttr;
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(reassociationAttr, parser.getBuilder().getType<::mlir::NoneType>(), "reassociation",
          result.attributes)) {
    return ::mlir::failure();
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
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CollapseShapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getReassociationAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("reassociation");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CollapseShapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::CollapseShapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DeallocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DeallocOpGenericAdaptorBase::DeallocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.dealloc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DeallocOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr DeallocOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
DeallocOpAdaptor::DeallocOpAdaptor(DeallocOp op) : DeallocOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DeallocOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DeallocOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DeallocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> DeallocOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange DeallocOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DeallocOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DeallocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref) {
  odsState.addOperands(memref);
}

void DeallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeallocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DeallocOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DeallocOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DeallocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DeallocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void DeallocOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Free::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::DeallocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DimOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DimOpGenericAdaptorBase::DimOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.dim", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DimOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr DimOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
DimOpAdaptor::DimOpAdaptor(DimOp op) : DimOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DimOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DimOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DimOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DimOp::getSource() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> DimOp::getIndex() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange DimOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DimOp::getIndexMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DimOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DimOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> DimOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);
  odsState.addTypes(result);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(DimOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value index) {
  odsState.addOperands(source);
  odsState.addOperands(index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DimOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(DimOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult DimOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DimOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult DimOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult DimOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> indexOperands(indexRawOperands);  ::llvm::SMLoc indexOperandsLoc;
  (void)indexOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(indexRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indexOperands, odsBuildableType0, indexOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DimOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getIndex();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void DimOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::DimOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DmaStartOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DmaStartOpGenericAdaptorBase::DmaStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.dma_start", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DmaStartOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr DmaStartOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
DmaStartOpAdaptor::DmaStartOpAdaptor(DmaStartOp op) : DmaStartOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DmaStartOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DmaStartOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range DmaStartOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range DmaStartOp::getOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange DmaStartOp::getOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DmaStartOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DmaStartOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DmaStartOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void DmaStartOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DmaStartOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DmaStartOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::DmaStartOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DmaWaitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
DmaWaitOpGenericAdaptorBase::DmaWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.dma_wait", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> DmaWaitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr DmaWaitOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
DmaWaitOpAdaptor::DmaWaitOpAdaptor(DmaWaitOp op) : DmaWaitOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult DmaWaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DmaWaitOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range DmaWaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> DmaWaitOp::getTagMemRef() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range DmaWaitOp::getTagIndices() {
  return getODSOperands(1);
}

::mlir::TypedValue<::mlir::IndexType> DmaWaitOp::getNumElements() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(2).begin());
}

::mlir::MutableOperandRange DmaWaitOp::getTagMemRefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DmaWaitOp::getTagIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange DmaWaitOp::getNumElementsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DmaWaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DmaWaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DmaWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tagMemRef, ::mlir::ValueRange tagIndices, ::mlir::Value numElements) {
  odsState.addOperands(tagMemRef);
  odsState.addOperands(tagIndices);
  odsState.addOperands(numElements);
}

void DmaWaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tagMemRef, ::mlir::ValueRange tagIndices, ::mlir::Value numElements) {
  odsState.addOperands(tagMemRef);
  odsState.addOperands(tagIndices);
  odsState.addOperands(numElements);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DmaWaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DmaWaitOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DmaWaitOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult DmaWaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tagMemRefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tagMemRefOperands(tagMemRefRawOperands);  ::llvm::SMLoc tagMemRefOperandsLoc;
  (void)tagMemRefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> tagIndicesOperands;
  ::llvm::SMLoc tagIndicesOperandsLoc;
  (void)tagIndicesOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand numElementsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> numElementsOperands(numElementsRawOperands);  ::llvm::SMLoc numElementsOperandsLoc;
  (void)numElementsOperandsLoc;
  ::mlir::Type tagMemRefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tagMemRefTypes(tagMemRefRawTypes);

  tagMemRefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tagMemRefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  tagIndicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(tagIndicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  numElementsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(numElementsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tagMemRefRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(tagMemRefOperands, tagMemRefTypes, tagMemRefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(tagIndicesOperands, odsBuildableType0, tagIndicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(numElementsOperands, odsBuildableType0, numElementsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DmaWaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTagMemRef();
  _odsPrinter << "[";
  _odsPrinter << getTagIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getNumElements();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTagMemRef().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::DmaWaitOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExpandShapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExpandShapeOpGenericAdaptorBase::ExpandShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.expand_shape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExpandShapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExpandShapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExpandShapeOpGenericAdaptorBase::getReassociationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExpandShapeOp::getReassociationAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExpandShapeOpGenericAdaptorBase::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

} // namespace detail
ExpandShapeOpAdaptor::ExpandShapeOpAdaptor(ExpandShapeOp op) : ExpandShapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExpandShapeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.expand_shape' op ""requires attribute 'reassociation'");
    if (namedAttrIt->getName() == ExpandShapeOp::getReassociationAttrName(*odsOpName)) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_reassociation && !(((tblgen_reassociation.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reassociation.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); }))))
    return emitError(loc, "'memref.expand_shape' op ""attribute 'reassociation' failed to satisfy constraint: Array of 64-bit integer array attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExpandShapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpandShapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ExpandShapeOp::getSrc() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExpandShapeOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExpandShapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpandShapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ExpandShapeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::ArrayAttr ExpandShapeOp::getReassociationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReassociationAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExpandShapeOp::getReassociation() {
  auto attr = getReassociationAttr();
  return attr;
}

void ExpandShapeOp::setReassociationAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReassociationAttrName(), attr);
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState, resultType, src, attrs);
      odsState.addAttribute("reassociation",
                          getReassociationIndicesAttribute(odsBuilder, reassociation));
    
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs) {
      auto reassociationMaps =
          convertReassociationMapsToIndices(odsBuilder, reassociation);
      build(odsBuilder, odsState, resultType, src, reassociationMaps, attrs);
    
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  odsState.addTypes(result);
}

void ExpandShapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation) {
  odsState.addOperands(src);
  odsState.addAttribute(getReassociationAttrName(odsState.name), reassociation);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpandShapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExpandShapeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassociation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'reassociation'");
    if (namedAttrIt->getName() == getReassociationAttrName()) {
      tblgen_reassociation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps4(*this, tblgen_reassociation, "reassociation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExpandShapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExpandShapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::ArrayAttr reassociationAttr;
  ::mlir::Type srcRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> srcTypes(srcRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(reassociationAttr, parser.getBuilder().getType<::mlir::NoneType>(), "reassociation",
          result.attributes)) {
    return ::mlir::failure();
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
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(srcOperands, srcTypes, srcOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExpandShapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSrc();
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getReassociationAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("reassociation");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSrc().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExpandShapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ExpandShapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExtractAlignedPointerAsIndexOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractAlignedPointerAsIndexOpGenericAdaptorBase::ExtractAlignedPointerAsIndexOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.extract_aligned_pointer_as_index", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractAlignedPointerAsIndexOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExtractAlignedPointerAsIndexOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ExtractAlignedPointerAsIndexOpAdaptor::ExtractAlignedPointerAsIndexOpAdaptor(ExtractAlignedPointerAsIndexOp op) : ExtractAlignedPointerAsIndexOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractAlignedPointerAsIndexOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractAlignedPointerAsIndexOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractAlignedPointerAsIndexOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ExtractAlignedPointerAsIndexOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExtractAlignedPointerAsIndexOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractAlignedPointerAsIndexOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractAlignedPointerAsIndexOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> ExtractAlignedPointerAsIndexOp::getAlignedPointer() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

void ExtractAlignedPointerAsIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type aligned_pointer, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(aligned_pointer);
}

void ExtractAlignedPointerAsIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source) {
  odsState.addOperands(source);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ExtractAlignedPointerAsIndexOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ExtractAlignedPointerAsIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractAlignedPointerAsIndexOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExtractAlignedPointerAsIndexOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ExtractAlignedPointerAsIndexOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ExtractAlignedPointerAsIndexOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractAlignedPointerAsIndexOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ExtractAlignedPointerAsIndexOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ExtractAlignedPointerAsIndexOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractAlignedPointerAsIndexOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ExtractAlignedPointerAsIndexOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ExtractAlignedPointerAsIndexOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExtractStridedMetadataOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractStridedMetadataOpGenericAdaptorBase::ExtractStridedMetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.extract_strided_metadata", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractStridedMetadataOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExtractStridedMetadataOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ExtractStridedMetadataOpAdaptor::ExtractStridedMetadataOpAdaptor(ExtractStridedMetadataOp op) : ExtractStridedMetadataOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExtractStridedMetadataOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractStridedMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractStridedMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ExtractStridedMetadataOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExtractStridedMetadataOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractStridedMetadataOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 2) / 2;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ExtractStridedMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractStridedMetadataOp::getBaseBuffer() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> ExtractStridedMetadataOp::getOffset() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(1).begin());
}

::mlir::Operation::result_range ExtractStridedMetadataOp::getSizes() {
  return getODSResults(2);
}

::mlir::Operation::result_range ExtractStridedMetadataOp::getStrides() {
  return getODSResults(3);
}

void ExtractStridedMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type base_buffer, ::mlir::Type offset, ::mlir::TypeRange sizes, ::mlir::TypeRange strides, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(base_buffer);
  odsState.addTypes(offset);
  odsState.addTypes(sizes);
  odsState.addTypes(strides);
}

void ExtractStridedMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source) {
  odsState.addOperands(source);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ExtractStridedMetadataOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ExtractStridedMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() >= 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractStridedMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExtractStridedMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ExtractStridedMetadataOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() >= 2u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ExtractStridedMetadataOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSResults(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSResults(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractStridedMetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExtractStridedMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractStridedMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ExtractStridedMetadataOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ExtractStridedMetadataOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GetGlobalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetGlobalOpGenericAdaptorBase::GetGlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.get_global", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetGlobalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetGlobalOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr GetGlobalOpGenericAdaptorBase::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetGlobalOp::getNameAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::StringRef GetGlobalOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
GetGlobalOpAdaptor::GetGlobalOpAdaptor(GetGlobalOp op) : GetGlobalOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetGlobalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.get_global' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == GetGlobalOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'memref.get_global' op ""attribute 'name' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetGlobalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetGlobalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GetGlobalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetGlobalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> GetGlobalOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::FlatSymbolRefAttr GetGlobalOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNameAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef GetGlobalOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void GetGlobalOp::setNameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void GetGlobalOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void GetGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::FlatSymbolRefAttr name) {
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addTypes(result);
}

void GetGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr name) {
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name) {
  odsState.addAttribute(getNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), name));
  odsState.addTypes(result);
}

void GetGlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name) {
  odsState.addAttribute(getNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetGlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetGlobalOp::verifyInvariantsImpl() {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps5(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetGlobalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetGlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr nameAttr;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void GetGlobalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetGlobalOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::GetGlobalOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GlobalOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GlobalOpGenericAdaptorBase::GlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.global", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GlobalOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GlobalOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GlobalOpGenericAdaptorBase::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GlobalOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GlobalOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr GlobalOpGenericAdaptorBase::getSymVisibilityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GlobalOp::getSymVisibilityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > GlobalOpGenericAdaptorBase::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::TypeAttr GlobalOpGenericAdaptorBase::getTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GlobalOp::getTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::MemRefType GlobalOpGenericAdaptorBase::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::MemRefType>();
}

::mlir::Attribute GlobalOpGenericAdaptorBase::getInitialValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GlobalOp::getInitialValueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::Attribute>();
  return attr;
}

::std::optional<::mlir::Attribute> GlobalOpGenericAdaptorBase::getInitialValue() {
  auto attr = getInitialValueAttr();
  return attr ? ::std::optional<::mlir::Attribute>(attr) : (::std::nullopt);
}

::mlir::UnitAttr GlobalOpGenericAdaptorBase::getConstantAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GlobalOp::getConstantAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GlobalOpGenericAdaptorBase::getConstant() {
  auto attr = getConstantAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::IntegerAttr GlobalOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GlobalOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> GlobalOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GlobalOpAdaptor::GlobalOpAdaptor(GlobalOp op) : GlobalOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GlobalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_constant;
  ::mlir::Attribute tblgen_initial_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.global' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == GlobalOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getConstantAttrName(*odsOpName)) {
      tblgen_constant = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getInitialValueAttrName(*odsOpName)) {
      tblgen_initial_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.global' op ""requires attribute 'type'");
    if (namedAttrIt->getName() == GlobalOp::getTypeAttrName(*odsOpName)) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getSymVisibilityAttrName(*odsOpName)) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'memref.global' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_sym_visibility && !((tblgen_sym_visibility.isa<::mlir::StringAttr>())))
    return emitError(loc, "'memref.global' op ""attribute 'sym_visibility' failed to satisfy constraint: string attribute");

  if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::MemRefType>())) && ((true))))
    return emitError(loc, "'memref.global' op ""attribute 'type' failed to satisfy constraint: memref type attribute");

  if (tblgen_initial_value && !((true)))
    return emitError(loc, "'memref.global' op ""attribute 'initial_value' failed to satisfy constraint: any attribute");

  if (tblgen_constant && !((tblgen_constant.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'memref.global' op ""attribute 'constant' failed to satisfy constraint: unit attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'memref.global' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr GlobalOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GlobalOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr GlobalOp::getSymVisibilityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getSymVisibilityAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > GlobalOp::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::TypeAttr GlobalOp::getTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::MemRefType GlobalOp::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::MemRefType>();
}

::mlir::Attribute GlobalOp::getInitialValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getInitialValueAttrName()).dyn_cast_or_null<::mlir::Attribute>();
}

::std::optional<::mlir::Attribute> GlobalOp::getInitialValue() {
  auto attr = getInitialValueAttr();
  return attr ? ::std::optional<::mlir::Attribute>(attr) : (::std::nullopt);
}

::mlir::UnitAttr GlobalOp::getConstantAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getConstantAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getConstant() {
  auto attr = getConstantAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::IntegerAttr GlobalOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> GlobalOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GlobalOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void GlobalOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void GlobalOp::setSymVisibilityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymVisibilityAttrName(), attr);
}

void GlobalOp::setSymVisibility(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getSymVisibilityAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getSymVisibilityAttrName());
}

void GlobalOp::setTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTypeAttrName(), attr);
}

void GlobalOp::setType(::mlir::MemRefType attrValue) {
  (*this)->setAttr(getTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void GlobalOp::setInitialValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getInitialValueAttrName(), attr);
}

void GlobalOp::setConstantAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getConstantAttrName(), attr);
}

void GlobalOp::setConstant(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getConstantAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getConstantAttrName());
}

void GlobalOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void GlobalOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute GlobalOp::removeSymVisibilityAttr() {
  return (*this)->removeAttr(getSymVisibilityAttrName());
}

::mlir::Attribute GlobalOp::removeInitialValueAttr() {
  return (*this)->removeAttr(getInitialValueAttrName());
}

::mlir::Attribute GlobalOp::removeConstantAttr() {
  return (*this)->removeAttr(getConstantAttrName());
}

::mlir::Attribute GlobalOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (sym_visibility) {
    odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  if (initial_value) {
    odsState.addAttribute(getInitialValueAttrName(odsState.name), initial_value);
  }
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (sym_visibility) {
    odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  if (initial_value) {
    odsState.addAttribute(getInitialValueAttrName(odsState.name), initial_value);
  }
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (sym_visibility) {
    odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  if (initial_value) {
    odsState.addAttribute(getInitialValueAttrName(odsState.name), initial_value);
  }
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), ((constant) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (sym_visibility) {
    odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  if (initial_value) {
    odsState.addAttribute(getInitialValueAttrName(odsState.name), initial_value);
  }
  if (constant) {
    odsState.addAttribute(getConstantAttrName(odsState.name), ((constant) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_constant;
  ::mlir::Attribute tblgen_initial_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getConstantAttrName()) {
      tblgen_constant = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getInitialValueAttrName()) {
      tblgen_initial_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'type'");
    if (namedAttrIt->getName() == getTypeAttrName()) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getSymVisibilityAttrName()) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps6(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps6(*this, tblgen_sym_visibility, "sym_visibility")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps7(*this, tblgen_type, "type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps8(*this, tblgen_initial_value, "initial_value")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps9(*this, tblgen_constant, "constant")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps10(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult GlobalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_visibilityAttr;
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::TypeAttr typeAttr;
  ::mlir::Attribute initial_valueAttr;

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(sym_visibilityAttr, parser.getBuilder().getType<::mlir::NoneType>(), "sym_visibility", result.attributes);
    if (parseResult.has_value() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (sym_visibilityAttr) {
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("constant"))) {
    result.addAttribute("constant", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  {
    if (parseGlobalMemrefOpTypeAndInitialValue(parser, typeAttr, initial_valueAttr))
      return ::mlir::failure();
    result.addAttribute("type", typeAttr);
    if (initial_valueAttr)
      result.addAttribute("initial_value", initial_valueAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("sym_visibility")) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getSymVisibilityAttr());
  }
  if ((*this)->getAttr("constant")) {
    _odsPrinter << ' ' << "constant";
  }
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  printGlobalMemrefOpTypeAndInitialValue(_odsPrinter, *this, getTypeAttr(), getInitialValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("sym_visibility");
  elidedAttrs.push_back("constant");
  elidedAttrs.push_back("sym_name");
  elidedAttrs.push_back("type");
  elidedAttrs.push_back("initial_value");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getConstantAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("constant");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::GlobalOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::MemorySpaceCastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MemorySpaceCastOpGenericAdaptorBase::MemorySpaceCastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.memory_space_cast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> MemorySpaceCastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MemorySpaceCastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
MemorySpaceCastOpAdaptor::MemorySpaceCastOpAdaptor(MemorySpaceCastOp op) : MemorySpaceCastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult MemorySpaceCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MemorySpaceCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemorySpaceCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> MemorySpaceCastOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange MemorySpaceCastOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemorySpaceCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemorySpaceCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> MemorySpaceCastOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSResults(0).begin());
}

void MemorySpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(dest);
}

void MemorySpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemorySpaceCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemorySpaceCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MemorySpaceCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MemorySpaceCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addTypes(destTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MemorySpaceCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MemorySpaceCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::MemorySpaceCastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::PrefetchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PrefetchOpGenericAdaptorBase::PrefetchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.prefetch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PrefetchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr PrefetchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr PrefetchOpGenericAdaptorBase::getIsWriteAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, PrefetchOp::getIsWriteAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool PrefetchOpGenericAdaptorBase::getIsWrite() {
  auto attr = getIsWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr PrefetchOpGenericAdaptorBase::getLocalityHintAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, PrefetchOp::getLocalityHintAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t PrefetchOpGenericAdaptorBase::getLocalityHint() {
  auto attr = getLocalityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr PrefetchOpGenericAdaptorBase::getIsDataCacheAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, PrefetchOp::getIsDataCacheAttrName(*odsOpName)).cast<::mlir::BoolAttr>();
  return attr;
}

bool PrefetchOpGenericAdaptorBase::getIsDataCache() {
  auto attr = getIsDataCacheAttr();
  return attr.getValue();
}

} // namespace detail
PrefetchOpAdaptor::PrefetchOpAdaptor(PrefetchOp op) : PrefetchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PrefetchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_isDataCache;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.prefetch' op ""requires attribute 'isDataCache'");
    if (namedAttrIt->getName() == PrefetchOp::getIsDataCacheAttrName(*odsOpName)) {
      tblgen_isDataCache = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isWrite;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.prefetch' op ""requires attribute 'isWrite'");
    if (namedAttrIt->getName() == PrefetchOp::getIsWriteAttrName(*odsOpName)) {
      tblgen_isWrite = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_localityHint;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.prefetch' op ""requires attribute 'localityHint'");
    if (namedAttrIt->getName() == PrefetchOp::getLocalityHintAttrName(*odsOpName)) {
      tblgen_localityHint = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_isWrite && !((tblgen_isWrite.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'memref.prefetch' op ""attribute 'isWrite' failed to satisfy constraint: bool attribute");

  if (tblgen_localityHint && !((((tblgen_localityHint.isa<::mlir::IntegerAttr>())) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() >= 0)) && ((tblgen_localityHint.cast<::mlir::IntegerAttr>().getInt() <= 3))))
    return emitError(loc, "'memref.prefetch' op ""attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");

  if (tblgen_isDataCache && !((tblgen_isDataCache.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'memref.prefetch' op ""attribute 'isDataCache' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PrefetchOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range PrefetchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> PrefetchOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range PrefetchOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange PrefetchOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PrefetchOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PrefetchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PrefetchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr PrefetchOp::getIsWriteAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIsWriteAttrName()).cast<::mlir::BoolAttr>();
}

bool PrefetchOp::getIsWrite() {
  auto attr = getIsWriteAttr();
  return attr.getValue();
}

::mlir::IntegerAttr PrefetchOp::getLocalityHintAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getLocalityHintAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t PrefetchOp::getLocalityHint() {
  auto attr = getLocalityHintAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr PrefetchOp::getIsDataCacheAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getIsDataCacheAttrName()).cast<::mlir::BoolAttr>();
}

bool PrefetchOp::getIsDataCache() {
  auto attr = getIsDataCacheAttr();
  return attr.getValue();
}

void PrefetchOp::setIsWriteAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getIsWriteAttrName(), attr);
}

void PrefetchOp::setIsWrite(bool attrValue) {
  (*this)->setAttr(getIsWriteAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void PrefetchOp::setLocalityHintAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLocalityHintAttrName(), attr);
}

void PrefetchOp::setLocalityHint(uint32_t attrValue) {
  (*this)->setAttr(getLocalityHintAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void PrefetchOp::setIsDataCacheAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getIsDataCacheAttrName(), attr);
}

void PrefetchOp::setIsDataCache(bool attrValue) {
  (*this)->setAttr(getIsDataCacheAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), isWrite);
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), localityHint);
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), isDataCache);
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), isWrite);
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), localityHint);
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), isDataCache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), odsBuilder.getBoolAttr(isDataCache));
}

void PrefetchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getIsWriteAttrName(odsState.name), odsBuilder.getBoolAttr(isWrite));
  odsState.addAttribute(getLocalityHintAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), localityHint));
  odsState.addAttribute(getIsDataCacheAttrName(odsState.name), odsBuilder.getBoolAttr(isDataCache));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrefetchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PrefetchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_isDataCache;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'isDataCache'");
    if (namedAttrIt->getName() == getIsDataCacheAttrName()) {
      tblgen_isDataCache = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isWrite;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'isWrite'");
    if (namedAttrIt->getName() == getIsWriteAttrName()) {
      tblgen_isWrite = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_localityHint;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'localityHint'");
    if (namedAttrIt->getName() == getLocalityHintAttrName()) {
      tblgen_localityHint = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps2(*this, tblgen_isWrite, "isWrite")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps11(*this, tblgen_localityHint, "localityHint")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps2(*this, tblgen_isDataCache, "isDataCache")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PrefetchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::PrefetchOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::RankOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RankOpGenericAdaptorBase::RankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.rank", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RankOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr RankOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
RankOpAdaptor::RankOpAdaptor(RankOp op) : RankOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult RankOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RankOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range RankOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> RankOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange RankOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> RankOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RankOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value memref) {
  odsState.addOperands(memref);
  odsState.addTypes(resultType0);
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref) {
  odsState.addOperands(memref);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(RankOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RankOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RankOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(RankOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult RankOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult RankOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult RankOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult RankOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RankOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void RankOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::RankOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReallocOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReallocOpGenericAdaptorBase::ReallocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.realloc", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReallocOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ReallocOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ReallocOpGenericAdaptorBase::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReallocOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint64_t> ReallocOpGenericAdaptorBase::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
ReallocOpAdaptor::ReallocOpAdaptor(ReallocOp op) : ReallocOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReallocOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ReallocOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !((((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'memref.realloc' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReallocOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReallocOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ReallocOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> ReallocOp::getDynamicResultSize() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IndexType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*operands.begin());
}

::mlir::MutableOperandRange ReallocOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReallocOp::getDynamicResultSizeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReallocOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReallocOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr ReallocOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint64_t> ReallocOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::std::optional<uint64_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void ReallocOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void ReallocOp::setAlignment(::std::optional<uint64_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getAlignmentAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), *attrValue));
    (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute ReallocOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

void ReallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, Value dynamicResultSize) {
      return build(odsBuilder, odsState, resultType, source, dynamicResultSize,
                   IntegerAttr());
    
}

void ReallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source, /*optional*/::mlir::Value dynamicResultSize, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(source);
  if (dynamicResultSize)
    odsState.addOperands(dynamicResultSize);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(resultType0);
}

void ReallocOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, /*optional*/::mlir::Value dynamicResultSize, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(source);
  if (dynamicResultSize)
    odsState.addOperands(dynamicResultSize);
  if (alignment) {
    odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReallocOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReallocOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps3(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReallocOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReallocOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> dynamicResultSizeOperands;
  ::llvm::SMLoc dynamicResultSizeOperandsLoc;
  (void)dynamicResultSizeOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  {
    dynamicResultSizeOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      dynamicResultSizeOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dynamicResultSizeOperands, odsBuildableType0, dynamicResultSizeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReallocOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  if (getDynamicResultSize()) {
    _odsPrinter << "(";
    if (::mlir::Value value = getDynamicResultSize())
      _odsPrinter << value;
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ReallocOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Free::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ReallocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReinterpretCastOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReinterpretCastOpGenericAdaptorBase::ReinterpretCastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.reinterpret_cast", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReinterpretCastOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ReinterpretCastOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr ReinterpretCastOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr ReinterpretCastOpGenericAdaptorBase::getStaticOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, ReinterpretCastOp::getStaticOffsetsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ReinterpretCastOpGenericAdaptorBase::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ReinterpretCastOpGenericAdaptorBase::getStaticSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ReinterpretCastOp::getStaticSizesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ReinterpretCastOpGenericAdaptorBase::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ReinterpretCastOpGenericAdaptorBase::getStaticStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, ReinterpretCastOp::getStaticStridesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> ReinterpretCastOpGenericAdaptorBase::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

} // namespace detail
ReinterpretCastOpAdaptor::ReinterpretCastOpAdaptor(ReinterpretCastOp op) : ReinterpretCastOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReinterpretCastOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.reinterpret_cast' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ReinterpretCastOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.reinterpret_cast' op ""requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == ReinterpretCastOp::getStaticOffsetsAttrName(*odsOpName)) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.reinterpret_cast' op ""requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == ReinterpretCastOp::getStaticSizesAttrName(*odsOpName)) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.reinterpret_cast' op ""requires attribute 'static_strides'");
    if (namedAttrIt->getName() == ReinterpretCastOp::getStaticStridesAttrName(*odsOpName)) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'memref.reinterpret_cast' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_offsets && !((tblgen_static_offsets.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'memref.reinterpret_cast' op ""attribute 'static_offsets' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_sizes && !((tblgen_static_sizes.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'memref.reinterpret_cast' op ""attribute 'static_sizes' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_strides && !((tblgen_static_strides.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'memref.reinterpret_cast' op ""attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReinterpretCastOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ReinterpretCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> ReinterpretCastOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range ReinterpretCastOp::getOffsets() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ReinterpretCastOp::getSizes() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ReinterpretCastOp::getStrides() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange ReinterpretCastOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReinterpretCastOp::getOffsetsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReinterpretCastOp::getSizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReinterpretCastOp::getStridesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ReinterpretCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReinterpretCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ReinterpretCastOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr ReinterpretCastOp::getStaticOffsetsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getStaticOffsetsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ReinterpretCastOp::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ReinterpretCastOp::getStaticSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getStaticSizesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ReinterpretCastOp::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr ReinterpretCastOp::getStaticStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getStaticStridesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> ReinterpretCastOp::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

void ReinterpretCastOp::setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticOffsetsAttrName(), attr);
}

void ReinterpretCastOp::setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticOffsetsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ReinterpretCastOp::setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticSizesAttrName(), attr);
}

void ReinterpretCastOp::setStaticSizes(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticSizesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ReinterpretCastOp::setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticStridesAttrName(), attr);
}

void ReinterpretCastOp::setStaticStrides(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticStridesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void ReinterpretCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  odsState.addTypes(result);
}

void ReinterpretCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReinterpretCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  odsState.addTypes(result);
}

void ReinterpretCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReinterpretCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReinterpretCastOp::verifyInvariantsImpl() {
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
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == getStaticOffsetsAttrName()) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == getStaticSizesAttrName()) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_strides'");
    if (namedAttrIt->getName() == getStaticStridesAttrName()) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps12(*this, tblgen_static_offsets, "static_offsets")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps12(*this, tblgen_static_sizes, "static_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps12(*this, tblgen_static_strides, "static_strides")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReinterpretCastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReinterpretCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_offsetsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_sizesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();
  if (parser.parseKeyword("offset"))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseKeyword("sizes"))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseKeyword("strides"))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, offsetsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, stridesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReinterpretCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ' << "offset";
  _odsPrinter << ":";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getOffsets(), getStaticOffsetsAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ' << "sizes";
  _odsPrinter << ":";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getSizes(), getStaticSizesAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ' << "strides";
  _odsPrinter << ":";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getStrides(), getStaticStridesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("static_offsets");
  elidedAttrs.push_back("static_sizes");
  elidedAttrs.push_back("static_strides");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ReinterpretCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ReinterpretCastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReshapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReshapeOpGenericAdaptorBase::ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.reshape", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReshapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReshapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp op) : ReshapeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> ReshapeOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> ReshapeOp::getShape() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange ReshapeOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::getShapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::BaseMemRefType> ReshapeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSResults(0).begin());
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value operand, Value shape) {
       odsState.addOperands(operand);
       odsState.addOperands(shape);
       odsState.addTypes(resultType);
     
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value shape) {
  odsState.addOperands(source);
  odsState.addOperands(shape);
  odsState.addTypes(result);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value shape) {
  odsState.addOperands(source);
  odsState.addOperands(shape);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReshapeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReshapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand shapeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> shapeOperands(shapeRawOperands);  ::llvm::SMLoc shapeOperandsLoc;
  (void)shapeOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  shapeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(shapeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(sourceOperands, shapeOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << "(";
  _odsPrinter << getShape();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

void ReshapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ReshapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::StoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StoreOpGenericAdaptorBase::StoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.store", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> StoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr StoreOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr StoreOpGenericAdaptorBase::getNontemporalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getNontemporalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  if (!attr)
    attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
  return attr;
}

bool StoreOpGenericAdaptorBase::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

} // namespace detail
StoreOpAdaptor::StoreOpAdaptor(StoreOp op) : StoreOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_nontemporal;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StoreOp::getNontemporalAttrName(*odsOpName)) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_nontemporal && !((tblgen_nontemporal.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'memref.store' op ""attribute 'nontemporal' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> StoreOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range StoreOp::getIndices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange StoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr StoreOp::getNontemporalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNontemporalAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool StoreOp::getNontemporal() {
  auto attr = getNontemporalAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void StoreOp::setNontemporalAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getNontemporalAttrName(), attr);
}

void StoreOp::setNontemporal(::std::optional<bool> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNontemporalAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(*attrValue));
    (*this)->removeAttr(getNontemporalAttrName());
}

::mlir::Attribute StoreOp::removeNontemporalAttr() {
  return (*this)->removeAttr(getNontemporalAttrName());
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref) {
      odsState.addOperands(valueToStore);
      odsState.addOperands(memref);
    
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (nontemporal) {
    odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getBoolAttr(nontemporal));
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getBoolAttr(nontemporal));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_nontemporal;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNontemporalAttrName()) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps2(*this, tblgen_nontemporal, "nontemporal")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<MemRefType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that type of 'value' matches element type of 'memref'");
  return ::mlir::success();
}

::mlir::LogicalResult StoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
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
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(valueOperands, memrefTypes[0].cast<MemRefType>().getElementType(), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNontemporalAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("nontemporal");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void StoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::StoreOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TransposeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.transpose", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TransposeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TransposeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr TransposeOpGenericAdaptorBase::getPermutationAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TransposeOp::getPermutationAttrName(*odsOpName)).cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap TransposeOpGenericAdaptorBase::getPermutation() {
  auto attr = getPermutationAttr();
  return attr.getValue();
}

} // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op) : TransposeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_permutation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.transpose' op ""requires attribute 'permutation'");
    if (namedAttrIt->getName() == TransposeOp::getPermutationAttrName(*odsOpName)) {
      tblgen_permutation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_permutation && !((tblgen_permutation.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'memref.transpose' op ""attribute 'permutation' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> TransposeOp::getIn() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange TransposeOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::AffineMapAttr TransposeOp::getPermutationAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPermutationAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap TransposeOp::getPermutation() {
  auto attr = getPermutationAttr();
  return attr.getValue();
}

void TransposeOp::setPermutationAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getPermutationAttrName(), attr);
}

void TransposeOp::setPermutation(::mlir::AffineMap attrValue) {
  (*this)->setAttr(getPermutationAttrName(), ::mlir::AffineMapAttr::get(attrValue));
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMapAttr permutation) {
  odsState.addOperands(in);
  odsState.addAttribute(getPermutationAttrName(odsState.name), permutation);
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMapAttr permutation) {
  odsState.addOperands(in);
  odsState.addAttribute(getPermutationAttrName(odsState.name), permutation);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMap permutation) {
  odsState.addOperands(in);
  odsState.addAttribute(getPermutationAttrName(odsState.name), ::mlir::AffineMapAttr::get(permutation));
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMap permutation) {
  odsState.addOperands(in);
  odsState.addAttribute(getPermutationAttrName(odsState.name), ::mlir::AffineMapAttr::get(permutation));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_permutation;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'permutation'");
    if (namedAttrIt->getName() == getPermutationAttrName()) {
      tblgen_permutation = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps13(*this, tblgen_permutation, "permutation")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransposeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void TransposeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::TransposeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ViewOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ViewOpGenericAdaptorBase::ViewOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.view", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ViewOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ViewOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ViewOpAdaptor::ViewOpAdaptor(ViewOp op) : ViewOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ViewOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ViewOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ViewOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> ViewOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> ViewOp::getByteShift() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range ViewOp::getSizes() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ViewOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ViewOp::getByteShiftMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ViewOp::getSizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ViewOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ViewOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes) {
  odsState.addOperands(source);
  odsState.addOperands(byte_shift);
  odsState.addOperands(sizes);
  odsState.addTypes(resultType0);
}

void ViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes) {
  odsState.addOperands(source);
  odsState.addOperands(byte_shift);
  odsState.addOperands(sizes);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ViewOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ViewOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ViewOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ViewOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand byte_shiftRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> byte_shiftOperands(byte_shiftRawOperands);  ::llvm::SMLoc byte_shiftOperandsLoc;
  (void)byte_shiftOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  byte_shiftOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(byte_shiftRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  sizesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(sizesOperands))
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
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(byte_shiftOperands, odsBuildableType0, byte_shiftOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ViewOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << "[";
  _odsPrinter << getByteShift();
  _odsPrinter << "]";
  _odsPrinter << "[";
  _odsPrinter << getSizes();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ViewOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::ViewOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::SubViewOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SubViewOpGenericAdaptorBase::SubViewOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.subview", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SubViewOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, SubViewOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SubViewOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseI64ArrayAttr SubViewOpGenericAdaptorBase::getStaticOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, SubViewOp::getStaticOffsetsAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> SubViewOpGenericAdaptorBase::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr SubViewOpGenericAdaptorBase::getStaticSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, SubViewOp::getStaticSizesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> SubViewOpGenericAdaptorBase::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr SubViewOpGenericAdaptorBase::getStaticStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, SubViewOp::getStaticStridesAttrName(*odsOpName)).cast<::mlir::DenseI64ArrayAttr>();
  return attr;
}

::llvm::ArrayRef<int64_t> SubViewOpGenericAdaptorBase::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

} // namespace detail
SubViewOpAdaptor::SubViewOpAdaptor(SubViewOp op) : SubViewOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SubViewOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.subview' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SubViewOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.subview' op ""requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == SubViewOp::getStaticOffsetsAttrName(*odsOpName)) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.subview' op ""requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == SubViewOp::getStaticSizesAttrName(*odsOpName)) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'memref.subview' op ""requires attribute 'static_strides'");
    if (namedAttrIt->getName() == SubViewOp::getStaticStridesAttrName(*odsOpName)) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'memref.subview' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_offsets && !((tblgen_static_offsets.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'memref.subview' op ""attribute 'static_offsets' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_sizes && !((tblgen_static_sizes.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'memref.subview' op ""attribute 'static_sizes' failed to satisfy constraint: i64 dense array attribute");

  if (tblgen_static_strides && !((tblgen_static_strides.isa<::mlir::DenseI64ArrayAttr>())))
    return emitError(loc, "'memref.subview' op ""attribute 'static_strides' failed to satisfy constraint: i64 dense array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SubViewOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SubViewOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> SubViewOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range SubViewOp::getOffsets() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SubViewOp::getSizes() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range SubViewOp::getStrides() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange SubViewOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SubViewOp::getOffsetsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SubViewOp::getSizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SubViewOp::getStridesMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SubViewOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubViewOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> SubViewOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::DenseI64ArrayAttr SubViewOp::getStaticOffsetsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getStaticOffsetsAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> SubViewOp::getStaticOffsets() {
  auto attr = getStaticOffsetsAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr SubViewOp::getStaticSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getStaticSizesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> SubViewOp::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::DenseI64ArrayAttr SubViewOp::getStaticStridesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getStaticStridesAttrName()).cast<::mlir::DenseI64ArrayAttr>();
}

::llvm::ArrayRef<int64_t> SubViewOp::getStaticStrides() {
  auto attr = getStaticStridesAttr();
  return attr;
}

void SubViewOp::setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticOffsetsAttrName(), attr);
}

void SubViewOp::setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticOffsetsAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void SubViewOp::setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticSizesAttrName(), attr);
}

void SubViewOp::setStaticSizes(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticSizesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void SubViewOp::setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr) {
  (*this)->setAttr(getStaticStridesAttrName(), attr);
}

void SubViewOp::setStaticStrides(::llvm::ArrayRef<int64_t> attrValue) {
  (*this)->setAttr(getStaticStridesAttrName(), ::mlir::Builder((*this)->getContext()).getDenseI64ArrayAttr(attrValue));
}

void SubViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  odsState.addTypes(result);
}

void SubViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), static_offsets);
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), static_strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  odsState.addTypes(result);
}

void SubViewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addAttribute(getStaticOffsetsAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_offsets));
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_sizes));
  odsState.addAttribute(getStaticStridesAttrName(odsState.name), odsBuilder.getDenseI64ArrayAttr(static_strides));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubViewOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubViewOp::verifyInvariantsImpl() {
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
  ::mlir::Attribute tblgen_static_offsets;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_offsets'");
    if (namedAttrIt->getName() == getStaticOffsetsAttrName()) {
      tblgen_static_offsets = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_sizes'");
    if (namedAttrIt->getName() == getStaticSizesAttrName()) {
      tblgen_static_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_strides;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_strides'");
    if (namedAttrIt->getName() == getStaticStridesAttrName()) {
      tblgen_static_strides = namedAttrIt->getValue();
      break;
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps12(*this, tblgen_static_offsets, "static_offsets")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps12(*this, tblgen_static_sizes, "static_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MemRefOps12(*this, tblgen_static_strides, "static_strides")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SubViewOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SubViewOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  (void)offsetsOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_offsetsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sizesOperands;
  ::llvm::SMLoc sizesOperandsLoc;
  (void)sizesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_sizesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  (void)stridesOperandsLoc;
  ::mlir::DenseI64ArrayAttr static_stridesAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
      return ::mlir::failure();
    result.addAttribute("static_offsets", static_offsetsAttr);
  }
  {
    sizesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, sizesOperands, static_sizesAttr))
      return ::mlir::failure();
    result.addAttribute("static_sizes", static_sizesAttr);
  }
  {
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, stridesOperands, static_stridesAttr))
      return ::mlir::failure();
    result.addAttribute("static_strides", static_stridesAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, offsetsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, stridesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubViewOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  printDynamicIndexList(_odsPrinter, *this, getOffsets(), getStaticOffsetsAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getSizes(), getStaticSizesAttr());
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getStrides(), getStaticStridesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("static_offsets");
  elidedAttrs.push_back("static_sizes");
  elidedAttrs.push_back("static_strides");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SubViewOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::SubViewOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TensorStoreOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TensorStoreOpGenericAdaptorBase::TensorStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("memref.tensor_store", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TensorStoreOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TensorStoreOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TensorStoreOpAdaptor::TensorStoreOpAdaptor(TensorStoreOp op) : TensorStoreOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TensorStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TensorStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TensorStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TensorStoreOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::BaseMemRefType> TensorStoreOp::getMemref() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::BaseMemRefType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange TensorStoreOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TensorStoreOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TensorStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TensorStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TensorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::Value memref) {
  odsState.addOperands(tensor);
  odsState.addOperands(memref);
}

void TensorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::Value memref) {
  odsState.addOperands(tensor);
  odsState.addOperands(memref);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TensorStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MemRefOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(getTensorTypeFromMemRefType((*this->getODSOperands(1).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that type of 'value' matches tensor equivalent of 'memref'");
  return ::mlir::success();
}

::mlir::LogicalResult TensorStoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TensorStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::BaseMemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  for (::mlir::Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<::mlir::BaseMemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be ranked or unranked memref of any type values, but got " << type;
    }
  }
  if (parser.resolveOperands(tensorOperands, getTensorTypeFromMemRefType(memrefTypes[0]), tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TensorStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TensorStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace memref
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::memref::TensorStoreOp)


#endif  // GET_OP_CLASSES

