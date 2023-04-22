/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::sparse_tensor::BinaryOp,
::mlir::sparse_tensor::CompressOp,
::mlir::sparse_tensor::ConcatenateOp,
::mlir::sparse_tensor::ConvertOp,
::mlir::sparse_tensor::ExpandOp,
::mlir::sparse_tensor::ForeachOp,
::mlir::sparse_tensor::GetStorageSpecifierOp,
::mlir::sparse_tensor::InsertOp,
::mlir::sparse_tensor::LoadOp,
::mlir::sparse_tensor::NewOp,
::mlir::sparse_tensor::NumberOfEntriesOp,
::mlir::sparse_tensor::OutOp,
::mlir::sparse_tensor::PackOp,
::mlir::sparse_tensor::PushBackOp,
::mlir::sparse_tensor::ReduceOp,
::mlir::sparse_tensor::SelectOp,
::mlir::sparse_tensor::SetStorageSpecifierOp,
::mlir::sparse_tensor::SortCooOp,
::mlir::sparse_tensor::SortOp,
::mlir::sparse_tensor::StorageSpecifierInitOp,
::mlir::sparse_tensor::ToCoordinatesBufferOp,
::mlir::sparse_tensor::ToCoordinatesOp,
::mlir::sparse_tensor::ToPositionsOp,
::mlir::sparse_tensor::ToSliceOffsetOp,
::mlir::sparse_tensor::ToSliceStrideOp,
::mlir::sparse_tensor::ToValuesOp,
::mlir::sparse_tensor::UnaryOp,
::mlir::sparse_tensor::UnpackOp,
::mlir::sparse_tensor::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace sparse_tensor {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (( isStrided(type.cast<::mlir::MemRefType>()) ))) && ((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be strided memref of any type values of rank 1, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (elementType.isa<::mlir::IndexType>()); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((!!::mlir::sparse_tensor::getSparseTensorEncoding(type)))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be sparse tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::RankedTensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::sparse_tensor::StorageSpecifierType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be metadata, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of signless integer or index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::IntegerType>())) || ((elementType.isa<::mlir::IndexType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D memref of integer or index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((!!::mlir::sparse_tensor::getSparseTensorEncoding(type) &&   ::mlir::sparse_tensor::getSparseTensorEncoding(type).isSlice()))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be sparse tensor slice of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::RankedTensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 1)))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1D tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::RankedTensorType>())) && (((type.cast<::mlir::ShapedType>().hasRank())) && ((type.cast<::mlir::ShapedType>().getRank()
                         == 2)))) && ([](::mlir::Type elementType) { return (elementType.isSignlessIntOrIndex()); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 2D tensor of signless integer or index values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SparseTensorOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessIntOrIndex()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: dimension attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::AffineMapAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::sparse_tensor::StorageSpecifierKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: sparse tensor storage specifier kind";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: level attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: index attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SparseTensorOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::sparse_tensor::SparseTensorSortKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: sparse tensor sort algorithm";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_SparseTensorOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_SparseTensorOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace sparse_tensor
} // namespace mlir
namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::BinaryOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BinaryOpGenericAdaptorBase::BinaryOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.binary", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BinaryOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BinaryOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr BinaryOpGenericAdaptorBase::getLeftIdentityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BinaryOp::getLeftIdentityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool BinaryOpGenericAdaptorBase::getLeftIdentity() {
  auto attr = getLeftIdentityAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr BinaryOpGenericAdaptorBase::getRightIdentityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, BinaryOp::getRightIdentityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool BinaryOpGenericAdaptorBase::getRightIdentity() {
  auto attr = getRightIdentityAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::Region &BinaryOpGenericAdaptorBase::getOverlapRegion() {
  return *odsRegions[0];
}

::mlir::Region &BinaryOpGenericAdaptorBase::getLeftRegion() {
  return *odsRegions[1];
}

::mlir::Region &BinaryOpGenericAdaptorBase::getRightRegion() {
  return *odsRegions[2];
}

::mlir::RegionRange BinaryOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
BinaryOpAdaptor::BinaryOpAdaptor(BinaryOp op) : BinaryOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult BinaryOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_left_identity;
  ::mlir::Attribute tblgen_right_identity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == BinaryOp::getLeftIdentityAttrName(*odsOpName)) {
      tblgen_left_identity = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == BinaryOp::getRightIdentityAttrName(*odsOpName)) {
      tblgen_right_identity = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_left_identity && !((tblgen_left_identity.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'sparse_tensor.binary' op ""attribute 'left_identity' failed to satisfy constraint: unit attribute");

  if (tblgen_right_identity && !((tblgen_right_identity.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'sparse_tensor.binary' op ""attribute 'right_identity' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BinaryOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BinaryOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BinaryOp::getX() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value BinaryOp::getY() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange BinaryOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BinaryOp::getYMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BinaryOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BinaryOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BinaryOp::getOutput() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Region &BinaryOp::getOverlapRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &BinaryOp::getLeftRegion() {
  return (*this)->getRegion(1);
}

::mlir::Region &BinaryOp::getRightRegion() {
  return (*this)->getRegion(2);
}

::mlir::UnitAttr BinaryOp::getLeftIdentityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLeftIdentityAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool BinaryOp::getLeftIdentity() {
  auto attr = getLeftIdentityAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::UnitAttr BinaryOp::getRightIdentityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getRightIdentityAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool BinaryOp::getRightIdentity() {
  auto attr = getRightIdentityAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void BinaryOp::setLeftIdentityAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getLeftIdentityAttrName(), attr);
}

void BinaryOp::setLeftIdentity(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getLeftIdentityAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getLeftIdentityAttrName());
}

void BinaryOp::setRightIdentityAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getRightIdentityAttrName(), attr);
}

void BinaryOp::setRightIdentity(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getRightIdentityAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getRightIdentityAttrName());
}

::mlir::Attribute BinaryOp::removeLeftIdentityAttr() {
  return (*this)->removeAttr(getLeftIdentityAttrName());
}

::mlir::Attribute BinaryOp::removeRightIdentityAttr() {
  return (*this)->removeAttr(getRightIdentityAttrName());
}

void BinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value x, ::mlir::Value y, /*optional*/::mlir::UnitAttr left_identity, /*optional*/::mlir::UnitAttr right_identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  if (left_identity) {
    odsState.addAttribute(getLeftIdentityAttrName(odsState.name), left_identity);
  }
  if (right_identity) {
    odsState.addAttribute(getRightIdentityAttrName(odsState.name), right_identity);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void BinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value y, /*optional*/::mlir::UnitAttr left_identity, /*optional*/::mlir::UnitAttr right_identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  if (left_identity) {
    odsState.addAttribute(getLeftIdentityAttrName(odsState.name), left_identity);
  }
  if (right_identity) {
    odsState.addAttribute(getRightIdentityAttrName(odsState.name), right_identity);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value x, ::mlir::Value y, /*optional*/bool left_identity, /*optional*/bool right_identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  if (left_identity) {
    odsState.addAttribute(getLeftIdentityAttrName(odsState.name), ((left_identity) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (right_identity) {
    odsState.addAttribute(getRightIdentityAttrName(odsState.name), ((right_identity) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void BinaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value y, /*optional*/bool left_identity, /*optional*/bool right_identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  if (left_identity) {
    odsState.addAttribute(getLeftIdentityAttrName(odsState.name), ((left_identity) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (right_identity) {
    odsState.addAttribute(getRightIdentityAttrName(odsState.name), ((right_identity) ? odsBuilder.getUnitAttr() : nullptr));
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BinaryOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 3; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BinaryOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_left_identity;
  ::mlir::Attribute tblgen_right_identity;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getLeftIdentityAttrName()) {
      tblgen_left_identity = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getRightIdentityAttrName()) {
      tblgen_right_identity = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps0(*this, tblgen_left_identity, "left_identity")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps0(*this, tblgen_right_identity, "right_identity")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps0(*this, region, "overlapRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps0(*this, region, "leftRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(2)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps0(*this, region, "rightRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult BinaryOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BinaryOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand yRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> yOperands(yRawOperands);  ::llvm::SMLoc yOperandsLoc;
  (void)yOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  ::mlir::Type yRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> yTypes(yRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  std::unique_ptr<::mlir::Region> overlapRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> leftRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> rightRegionRegion = std::make_unique<::mlir::Region>();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  yOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(yRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    yRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawTypes[0] = type;
  }
  if (parser.parseKeyword("overlap"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseRegion(*overlapRegionRegion))
    return ::mlir::failure();
  if (parser.parseKeyword("left"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("identity"))) {
    result.addAttribute("left_identity", parser.getBuilder().getUnitAttr());
  } else {

  if (parser.parseRegion(*leftRegionRegion))
    return ::mlir::failure();
  }
  if (parser.parseKeyword("right"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("identity"))) {
    result.addAttribute("right_identity", parser.getBuilder().getUnitAttr());
  } else {

  if (parser.parseRegion(*rightRegionRegion))
    return ::mlir::failure();
  }
  result.addRegion(std::move(overlapRegionRegion));
  result.addRegion(std::move(leftRegionRegion));
  result.addRegion(std::move(rightRegionRegion));
  result.addTypes(outputTypes);
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(yOperands, yTypes, yOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BinaryOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getY();
  _odsPrinter << ' ' << ":";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("left_identity");
  elidedAttrs.push_back("right_identity");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getLeftIdentityAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("left_identity");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getRightIdentityAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("right_identity");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getY().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printNewline();
  _odsPrinter << ' ' << "overlap";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getOverlapRegion());
  _odsPrinter.printNewline();
  _odsPrinter << ' ' << "left";
  _odsPrinter << ' ' << "=";
  if ((*this)->getAttr("left_identity")) {
    _odsPrinter << ' ' << "identity";
  } else {
    _odsPrinter << ' ';
    _odsPrinter.printRegion(getLeftRegion());
  }
  _odsPrinter.printNewline();
  _odsPrinter << ' ' << "right";
  _odsPrinter << ' ' << "=";
  if ((*this)->getAttr("right_identity")) {
    _odsPrinter << ' ' << "identity";
  } else {
    _odsPrinter << ' ';
    _odsPrinter.printRegion(getRightRegion());
  }
}

void BinaryOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::BinaryOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::CompressOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CompressOpGenericAdaptorBase::CompressOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.compress", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CompressOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 5) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr CompressOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CompressOpAdaptor::CompressOpAdaptor(CompressOp op) : CompressOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult CompressOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CompressOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 5) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range CompressOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CompressOp::getValues() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> CompressOp::getFilled() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::TypedValue<::mlir::MemRefType> CompressOp::getAdded() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(2).begin());
}

::mlir::TypedValue<::mlir::IndexType> CompressOp::getCount() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(3).begin());
}

::mlir::TypedValue<::mlir::TensorType> CompressOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(4).begin());
}

::mlir::Operation::operand_range CompressOp::getLvlCoords() {
  return getODSOperands(5);
}

::mlir::MutableOperandRange CompressOp::getValuesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressOp::getFilledMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressOp::getAddedMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressOp::getCountMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressOp::getLvlCoordsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CompressOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CompressOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CompressOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value values, ::mlir::Value filled, ::mlir::Value added, ::mlir::Value count, ::mlir::Value tensor, ::mlir::ValueRange lvlCoords) {
  odsState.addOperands(values);
  odsState.addOperands(filled);
  odsState.addOperands(added);
  odsState.addOperands(count);
  odsState.addOperands(tensor);
  odsState.addOperands(lvlCoords);
  odsState.addTypes(result);
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value values, ::mlir::Value filled, ::mlir::Value added, ::mlir::Value count, ::mlir::Value tensor, ::mlir::ValueRange lvlCoords) {
  odsState.addOperands(values);
  odsState.addOperands(filled);
  odsState.addOperands(added);
  odsState.addOperands(count);
  odsState.addOperands(tensor);
  odsState.addOperands(lvlCoords);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CompressOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values, ::mlir::Value filled, ::mlir::Value added, ::mlir::Value count, ::mlir::Value tensor, ::mlir::ValueRange lvlCoords) {
  odsState.addOperands(values);
  odsState.addOperands(filled);
  odsState.addOperands(added);
  odsState.addOperands(count);
  odsState.addOperands(tensor);
  odsState.addOperands(lvlCoords);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CompressOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CompressOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(CompressOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult CompressOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(4).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(4).begin()).getType()))))
    return emitOpError("failed to verify that all of {tensor, result} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult CompressOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult CompressOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[4].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult CompressOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valuesRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valuesOperands(valuesRawOperands);  ::llvm::SMLoc valuesOperandsLoc;
  (void)valuesOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand filledRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> filledOperands(filledRawOperands);  ::llvm::SMLoc filledOperandsLoc;
  (void)filledOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand addedRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addedOperands(addedRawOperands);  ::llvm::SMLoc addedOperandsLoc;
  (void)addedOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand countRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> countOperands(countRawOperands);  ::llvm::SMLoc countOperandsLoc;
  (void)countOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lvlCoordsOperands;
  ::llvm::SMLoc lvlCoordsOperandsLoc;
  (void)lvlCoordsOperandsLoc;
  ::mlir::Type valuesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valuesTypes(valuesRawTypes);
  ::mlir::Type filledRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> filledTypes(filledRawTypes);
  ::mlir::Type addedRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> addedTypes(addedRawTypes);
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  valuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valuesRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  filledOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(filledRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  addedOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addedRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  countOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(countRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  lvlCoordsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(lvlCoordsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valuesRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    filledRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addedRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(tensorTypes[0]);
  if (parser.resolveOperands(valuesOperands, valuesTypes, valuesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(filledOperands, filledTypes, filledOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(addedOperands, addedTypes, addedOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(countOperands, odsBuildableType0, countOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(lvlCoordsOperands, odsBuildableType0, lvlCoordsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CompressOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValues();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFilled();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getAdded();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCount();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  _odsPrinter << "[";
  _odsPrinter << getLvlCoords();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValues().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getFilled().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getAdded().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::CompressOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ConcatenateOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConcatenateOpGenericAdaptorBase::ConcatenateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.concatenate", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ConcatenateOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ConcatenateOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ConcatenateOpGenericAdaptorBase::getDimensionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConcatenateOp::getDimensionAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::sparse_tensor::Dimension ConcatenateOpGenericAdaptorBase::getDimension() {
  auto attr = getDimensionAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ConcatenateOpAdaptor::ConcatenateOpAdaptor(ConcatenateOp op) : ConcatenateOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ConcatenateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimension;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.concatenate' op ""requires attribute 'dimension'");
    if (namedAttrIt->getName() == ConcatenateOp::getDimensionAttrName(*odsOpName)) {
      tblgen_dimension = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dimension && !(((tblgen_dimension.isa<::mlir::IntegerAttr>())) && ((tblgen_dimension.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.concatenate' op ""attribute 'dimension' failed to satisfy constraint: dimension attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConcatenateOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ConcatenateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ConcatenateOp::getInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ConcatenateOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConcatenateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConcatenateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> ConcatenateOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr ConcatenateOp::getDimensionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDimensionAttrName()).cast<::mlir::IntegerAttr>();
}

::mlir::sparse_tensor::Dimension ConcatenateOp::getDimension() {
  auto attr = getDimensionAttr();
  return attr.getValue().getZExtValue();
}

void ConcatenateOp::setDimensionAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDimensionAttrName(), attr);
}

void ConcatenateOp::setDimension(::mlir::sparse_tensor::Dimension attrValue) {
  (*this)->setAttr(getDimensionAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), attrValue));
}

void ConcatenateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange inputs, ::mlir::IntegerAttr dimension) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getDimensionAttrName(odsState.name), dimension);
  odsState.addTypes(result);
}

void ConcatenateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::IntegerAttr dimension) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getDimensionAttrName(odsState.name), dimension);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatenateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange inputs, ::mlir::sparse_tensor::Dimension dimension) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dimension));
  odsState.addTypes(result);
}

void ConcatenateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::sparse_tensor::Dimension dimension) {
  odsState.addOperands(inputs);
  odsState.addAttribute(getDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dimension));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatenateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConcatenateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimension;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dimension'");
    if (namedAttrIt->getName() == getDimensionAttrName()) {
      tblgen_dimension = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps1(*this, tblgen_dimension, "dimension")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConcatenateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ConcatenateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConcatenateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputs();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getInputs().getTypes();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ConcatenateOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ConcatenateOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ConvertOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConvertOpGenericAdaptorBase::ConvertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.convert", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ConvertOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ConvertOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ConvertOpAdaptor::ConvertOpAdaptor(ConvertOp op) : ConvertOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ConvertOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConvertOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConvertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConvertOp::getSource() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ConvertOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConvertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConvertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConvertOp::getDest() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

void ConvertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(dest);
}

void ConvertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConvertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConvertOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConvertOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ConvertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addTypes(destTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConvertOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ConvertOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ConvertOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ExpandOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExpandOpGenericAdaptorBase::ExpandOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.expand", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExpandOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExpandOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ExpandOpAdaptor::ExpandOpAdaptor(ExpandOp op) : ExpandOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ExpandOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ExpandOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "values");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "filled");
  auto resultGroup2 = getODSResults(2);
  if (!resultGroup2.empty())
    setNameFn(*resultGroup2.begin(), "added");
  auto resultGroup3 = getODSResults(3);
  if (!resultGroup3.empty())
    setNameFn(*resultGroup3.begin(), "count");
}

std::pair<unsigned, unsigned> ExpandOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ExpandOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExpandOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExpandOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpandOp::getValues() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> ExpandOp::getFilled() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(1).begin());
}

::mlir::TypedValue<::mlir::MemRefType> ExpandOp::getAdded() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(2).begin());
}

::mlir::TypedValue<::mlir::IndexType> ExpandOp::getCount() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(3).begin());
}

void ExpandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type values, ::mlir::Type filled, ::mlir::Type added, ::mlir::Type count, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(values);
  odsState.addTypes(filled);
  odsState.addTypes(added);
  odsState.addTypes(count);
}

void ExpandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 4u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 4u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExpandOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSResults(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSResults(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExpandOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExpandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type valuesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valuesTypes(valuesRawTypes);
  ::mlir::Type filledRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> filledTypes(filledRawTypes);
  ::mlir::Type addedRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> addedTypes(addedRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valuesRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    filledRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addedRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(valuesTypes);
  result.addTypes(filledTypes);
  result.addTypes(addedTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExpandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getValues().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getFilled().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getAdded().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ExpandOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ForeachOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ForeachOpGenericAdaptorBase::ForeachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.foreach", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ForeachOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr ForeachOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr ForeachOpGenericAdaptorBase::getOrderAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ForeachOp::getOrderAttrName(*odsOpName)).dyn_cast_or_null<::mlir::AffineMapAttr>();
  return attr;
}

::std::optional< ::mlir::AffineMap > ForeachOpGenericAdaptorBase::getOrder() {
  auto attr = getOrderAttr();
  return attr ? ::std::optional< ::mlir::AffineMap >(attr.getValue()) : (::std::nullopt);
}

::mlir::Region &ForeachOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ForeachOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ForeachOpAdaptor::ForeachOpAdaptor(ForeachOp op) : ForeachOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ForeachOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_order;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ForeachOp::getOrderAttrName(*odsOpName)) {
      tblgen_order = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_order && !((tblgen_order.isa<::mlir::AffineMapAttr>())))
    return emitError(loc, "'sparse_tensor.foreach' op ""attribute 'order' failed to satisfy constraint: AffineMap attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ForeachOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ForeachOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ForeachOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range ForeachOp::getInitArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ForeachOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ForeachOp::getInitArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ForeachOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ForeachOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ForeachOp::getResults() {
  return getODSResults(0);
}

::mlir::Region &ForeachOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::AffineMapAttr ForeachOp::getOrderAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOrderAttrName()).dyn_cast_or_null<::mlir::AffineMapAttr>();
}

::std::optional< ::mlir::AffineMap > ForeachOp::getOrder() {
  auto attr = getOrderAttr();
  return attr ? ::std::optional< ::mlir::AffineMap >(attr.getValue()) : (::std::nullopt);
}

void ForeachOp::setOrderAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getOrderAttrName(), attr);
}

void ForeachOp::setOrder(::std::optional<::mlir::AffineMap> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getOrderAttrName(), ::mlir::AffineMapAttr::get(*attrValue));
    (*this)->removeAttr(getOrderAttrName());
}

::mlir::Attribute ForeachOp::removeOrderAttr() {
  return (*this)->removeAttr(getOrderAttrName());
}

void ForeachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value tensor, AffineMapAttr order, function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)> bodyBuilder) {
      build(odsBuilder, odsState, tensor, ValueRange(), order, bodyBuilder);
    
}

void ForeachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value tensor, function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)> bodyBuilder) {
      build(odsBuilder, odsState, tensor, ValueRange(), nullptr, bodyBuilder);
    
}

void ForeachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value tensor, ValueRange iterArgs, function_ref<void(OpBuilder &, Location, ValueRange, Value, ValueRange)> bodyBuilder) {
      build(odsBuilder, odsState, tensor, iterArgs, nullptr, bodyBuilder);
    
}

void ForeachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value tensor, ::mlir::ValueRange initArgs, /*optional*/::mlir::AffineMapAttr order) {
  odsState.addOperands(tensor);
  odsState.addOperands(initArgs);
  if (order) {
    odsState.addAttribute(getOrderAttrName(odsState.name), order);
  }
  (void)odsState.addRegion();
  odsState.addTypes(results);
}

void ForeachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ForeachOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_order;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getOrderAttrName()) {
      tblgen_order = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps2(*this, tblgen_order, "order")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForeachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ForeachOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initArgsOperands;
  ::llvm::SMLoc initArgsOperandsLoc;
  (void)initArgsOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> initArgsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (parser.parseKeyword("in"))
    return ::mlir::failure();

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("init"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  initArgsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initArgsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseTypeList(initArgsTypes))
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalArrow())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();

  ensureTerminator(*regionRegion, parser.getBuilder(), result.location);
  result.addRegion(std::move(regionRegion));
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(initArgsOperands, initArgsTypes, initArgsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ForeachOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "in";
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  if (!getInitArgs().empty()) {
    _odsPrinter << ' ' << "init";
    _odsPrinter << "(";
    _odsPrinter << getInitArgs();
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getInitArgs().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << getInitArgs().getTypes();
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getRegion().empty() ? nullptr : getRegion().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ForeachOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::GetStorageSpecifierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetStorageSpecifierOpGenericAdaptorBase::GetStorageSpecifierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.storage_specifier.get", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetStorageSpecifierOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetStorageSpecifierOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::sparse_tensor::StorageSpecifierKindAttr GetStorageSpecifierOpGenericAdaptorBase::getSpecifierKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetStorageSpecifierOp::getSpecifierKindAttrName(*odsOpName)).cast<::mlir::sparse_tensor::StorageSpecifierKindAttr>();
  return attr;
}

::mlir::sparse_tensor::StorageSpecifierKind GetStorageSpecifierOpGenericAdaptorBase::getSpecifierKind() {
  auto attr = getSpecifierKindAttr();
  return attr.getValue();
}

::mlir::IntegerAttr GetStorageSpecifierOpGenericAdaptorBase::getLevelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GetStorageSpecifierOp::getLevelAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<::mlir::sparse_tensor::Level> GetStorageSpecifierOpGenericAdaptorBase::getLevel() {
  auto attr = getLevelAttr();
  return attr ? ::std::optional<::mlir::sparse_tensor::Level>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetStorageSpecifierOpAdaptor::GetStorageSpecifierOpAdaptor(GetStorageSpecifierOp op) : GetStorageSpecifierOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetStorageSpecifierOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_specifierKind;
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.storage_specifier.get' op ""requires attribute 'specifierKind'");
    if (namedAttrIt->getName() == GetStorageSpecifierOp::getSpecifierKindAttrName(*odsOpName)) {
      tblgen_specifierKind = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GetStorageSpecifierOp::getLevelAttrName(*odsOpName)) {
      tblgen_level = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_specifierKind && !((tblgen_specifierKind.isa<::mlir::sparse_tensor::StorageSpecifierKindAttr>())))
    return emitError(loc, "'sparse_tensor.storage_specifier.get' op ""attribute 'specifierKind' failed to satisfy constraint: sparse tensor storage specifier kind");

  if (tblgen_level && !(((tblgen_level.isa<::mlir::IntegerAttr>())) && ((tblgen_level.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.storage_specifier.get' op ""attribute 'level' failed to satisfy constraint: level attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetStorageSpecifierOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetStorageSpecifierOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType> GetStorageSpecifierOp::getSpecifier() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetStorageSpecifierOp::getSpecifierMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetStorageSpecifierOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetStorageSpecifierOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> GetStorageSpecifierOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

::mlir::sparse_tensor::StorageSpecifierKindAttr GetStorageSpecifierOp::getSpecifierKindAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSpecifierKindAttrName()).cast<::mlir::sparse_tensor::StorageSpecifierKindAttr>();
}

::mlir::sparse_tensor::StorageSpecifierKind GetStorageSpecifierOp::getSpecifierKind() {
  auto attr = getSpecifierKindAttr();
  return attr.getValue();
}

::mlir::IntegerAttr GetStorageSpecifierOp::getLevelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getLevelAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<::mlir::sparse_tensor::Level> GetStorageSpecifierOp::getLevel() {
  auto attr = getLevelAttr();
  return attr ? ::std::optional<::mlir::sparse_tensor::Level>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetStorageSpecifierOp::setSpecifierKindAttr(::mlir::sparse_tensor::StorageSpecifierKindAttr attr) {
  (*this)->setAttr(getSpecifierKindAttrName(), attr);
}

void GetStorageSpecifierOp::setSpecifierKind(::mlir::sparse_tensor::StorageSpecifierKind attrValue) {
  (*this)->setAttr(getSpecifierKindAttrName(), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void GetStorageSpecifierOp::setLevelAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLevelAttrName(), attr);
}

void GetStorageSpecifierOp::setLevel(::std::optional<::mlir::sparse_tensor::Level> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getLevelAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), *attrValue));
    (*this)->removeAttr(getLevelAttrName());
}

::mlir::Attribute GetStorageSpecifierOp::removeLevelAttr() {
  return (*this)->removeAttr(getLevelAttrName());
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKind, /*optional*/::mlir::IntegerAttr level) {
  odsState.addOperands(specifier);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), specifierKind);
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  odsState.addTypes(result);
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKind, /*optional*/::mlir::IntegerAttr level) {
  odsState.addOperands(specifier);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), specifierKind);
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(GetStorageSpecifierOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKind, /*optional*/::mlir::IntegerAttr level) {
  odsState.addOperands(specifier);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), specifierKind);
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKind specifierKind, /*optional*/::mlir::IntegerAttr level) {
  odsState.addOperands(specifier);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(odsBuilder.getContext(), specifierKind));
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  odsState.addTypes(result);
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKind specifierKind, /*optional*/::mlir::IntegerAttr level) {
  odsState.addOperands(specifier);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(odsBuilder.getContext(), specifierKind));
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(GetStorageSpecifierOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKind specifierKind, /*optional*/::mlir::IntegerAttr level) {
  odsState.addOperands(specifier);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(odsBuilder.getContext(), specifierKind));
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void GetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(GetStorageSpecifierOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult GetStorageSpecifierOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_specifierKind;
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'specifierKind'");
    if (namedAttrIt->getName() == getSpecifierKindAttrName()) {
      tblgen_specifierKind = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getLevelAttrName()) {
      tblgen_level = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps3(*this, tblgen_specifierKind, "specifierKind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps4(*this, tblgen_level, "level")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetStorageSpecifierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult GetStorageSpecifierOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult GetStorageSpecifierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand specifierRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> specifierOperands(specifierRawOperands);  ::llvm::SMLoc specifierOperandsLoc;
  (void)specifierOperandsLoc;
  ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKindAttr;
  ::mlir::IntegerAttr levelAttr;
  ::mlir::Type specifierRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> specifierTypes(specifierRawTypes);

  specifierOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(specifierRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(specifierKindAttr, ::mlir::Type{}, "specifierKind",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("at"))) {

  if (parser.parseCustomAttributeWithFallback(levelAttr, parser.getBuilder().getIndexType(), "level",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(specifierRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(specifierOperands, specifierTypes, specifierOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetStorageSpecifierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSpecifier();
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getSpecifierKindAttr());
  if ((*this)->getAttr("level")) {
    _odsPrinter << ' ' << "at";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getLevelAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("specifierKind");
  elidedAttrs.push_back("level");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getSpecifier().getType();
}

void GetStorageSpecifierOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::GetStorageSpecifierOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::InsertOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
InsertOpGenericAdaptorBase::InsertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.insert", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> InsertOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr InsertOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
InsertOpAdaptor::InsertOpAdaptor(InsertOp op) : InsertOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult InsertOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InsertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> InsertOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range InsertOp::getLvlCoords() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InsertOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::getLvlCoordsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> InsertOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::mlir::Value tensor, ::mlir::ValueRange lvlCoords) {
  odsState.addOperands(value);
  odsState.addOperands(tensor);
  odsState.addOperands(lvlCoords);
  odsState.addTypes(result);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value tensor, ::mlir::ValueRange lvlCoords) {
  odsState.addOperands(value);
  odsState.addOperands(tensor);
  odsState.addOperands(lvlCoords);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value tensor, ::mlir::ValueRange lvlCoords) {
  odsState.addOperands(value);
  odsState.addOperands(tensor);
  odsState.addOperands(lvlCoords);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<ShapedType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that value type matches element type of tensor");
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {tensor, result} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult InsertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lvlCoordsOperands;
  ::llvm::SMLoc lvlCoordsOperandsLoc;
  (void)lvlCoordsOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  lvlCoordsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(lvlCoordsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  for (::mlir::Type type : tensorTypes) {
    (void)type;
    if (!((((type.isa<::mlir::TensorType>())) && ((!!::mlir::sparse_tensor::getSparseTensorEncoding(type)))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'tensor' must be sparse tensor of any type values, but got " << type;
    }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(tensorTypes[0]);
  if (parser.resolveOperands(valueOperands, tensorTypes[0].cast<ShapedType>().getElementType(), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(lvlCoordsOperands, odsBuildableType0, lvlCoordsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  _odsPrinter << "[";
  _odsPrinter << getLvlCoords();
  _odsPrinter << "]";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::InsertOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::LoadOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoadOpGenericAdaptorBase::LoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.load", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoadOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoadOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr LoadOpGenericAdaptorBase::getHasInsertsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getHasInsertsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool LoadOpGenericAdaptorBase::getHasInserts() {
  auto attr = getHasInsertsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
LoadOpAdaptor::LoadOpAdaptor(LoadOp op) : LoadOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hasInserts;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoadOp::getHasInsertsAttrName(*odsOpName)) {
      tblgen_hasInserts = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hasInserts && !((tblgen_hasInserts.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'sparse_tensor.load' op ""attribute 'hasInserts' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> LoadOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoadOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
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

::mlir::TypedValue<::mlir::TensorType> LoadOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::UnitAttr LoadOp::getHasInsertsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getHasInsertsAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::getHasInserts() {
  auto attr = getHasInsertsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void LoadOp::setHasInsertsAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getHasInsertsAttrName(), attr);
}

void LoadOp::setHasInserts(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getHasInsertsAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getHasInsertsAttrName());
}

::mlir::Attribute LoadOp::removeHasInsertsAttr() {
  return (*this)->removeAttr(getHasInsertsAttrName());
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, /*optional*/::mlir::UnitAttr hasInserts) {
  odsState.addOperands(tensor);
  if (hasInserts) {
    odsState.addAttribute(getHasInsertsAttrName(odsState.name), hasInserts);
  }
  odsState.addTypes(result);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, /*optional*/::mlir::UnitAttr hasInserts) {
  odsState.addOperands(tensor);
  if (hasInserts) {
    odsState.addAttribute(getHasInsertsAttrName(odsState.name), hasInserts);
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

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, /*optional*/::mlir::UnitAttr hasInserts) {
  odsState.addOperands(tensor);
  if (hasInserts) {
    odsState.addAttribute(getHasInsertsAttrName(odsState.name), hasInserts);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, /*optional*/bool hasInserts) {
  odsState.addOperands(tensor);
  if (hasInserts) {
    odsState.addAttribute(getHasInsertsAttrName(odsState.name), ((hasInserts) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(result);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, /*optional*/bool hasInserts) {
  odsState.addOperands(tensor);
  if (hasInserts) {
    odsState.addAttribute(getHasInsertsAttrName(odsState.name), ((hasInserts) ? odsBuilder.getUnitAttr() : nullptr));
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

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, /*optional*/bool hasInserts) {
  odsState.addOperands(tensor);
  if (hasInserts) {
    odsState.addAttribute(getHasInsertsAttrName(odsState.name), ((hasInserts) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
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
  ::mlir::Attribute tblgen_hasInserts;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHasInsertsAttrName()) {
      tblgen_hasInserts = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps0(*this, tblgen_hasInserts, "hasInserts")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult LoadOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("hasInserts"))) {
    result.addAttribute("hasInserts", parser.getBuilder().getUnitAttr());
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  result.addTypes(tensorTypes[0]);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  if ((*this)->getAttr("hasInserts")) {
    _odsPrinter << ' ' << "hasInserts";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("hasInserts");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getHasInsertsAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("hasInserts");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::LoadOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::NewOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
NewOpGenericAdaptorBase::NewOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.new", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> NewOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr NewOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
NewOpAdaptor::NewOpAdaptor(NewOp op) : NewOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult NewOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NewOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NewOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NewOp::getSource() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange NewOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> NewOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NewOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> NewOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

void NewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(result);
}

void NewOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NewOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NewOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NewOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult NewOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

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
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NewOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void NewOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::NewOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::NumberOfEntriesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
NumberOfEntriesOpGenericAdaptorBase::NumberOfEntriesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.number_of_entries", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> NumberOfEntriesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr NumberOfEntriesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
NumberOfEntriesOpAdaptor::NumberOfEntriesOpAdaptor(NumberOfEntriesOp op) : NumberOfEntriesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult NumberOfEntriesOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NumberOfEntriesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NumberOfEntriesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> NumberOfEntriesOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange NumberOfEntriesOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> NumberOfEntriesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NumberOfEntriesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> NumberOfEntriesOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

void NumberOfEntriesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(result);
}

void NumberOfEntriesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor) {
  odsState.addOperands(tensor);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(NumberOfEntriesOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void NumberOfEntriesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NumberOfEntriesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void NumberOfEntriesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(NumberOfEntriesOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult NumberOfEntriesOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NumberOfEntriesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult NumberOfEntriesOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult NumberOfEntriesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void NumberOfEntriesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void NumberOfEntriesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::NumberOfEntriesOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::OutOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
OutOpGenericAdaptorBase::OutOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.out", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> OutOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr OutOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
OutOpAdaptor::OutOpAdaptor(OutOp op) : OutOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult OutOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OutOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OutOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> OutOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::Value OutOp::getDest() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange OutOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OutOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OutOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OutOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void OutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::Value dest) {
  odsState.addOperands(tensor);
  odsState.addOperands(dest);
}

void OutOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::Value dest) {
  odsState.addOperands(tensor);
  odsState.addOperands(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OutOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OutOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OutOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult OutOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OutOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::OutOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::PackOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PackOpGenericAdaptorBase::PackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.pack", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PackOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PackOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr PackOpGenericAdaptorBase::getBatchedLvlsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PackOp::getBatchedLvlsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional< ::llvm::APInt > PackOpGenericAdaptorBase::getBatchedLvls() {
  auto attr = getBatchedLvlsAttr();
  return attr ? ::std::optional< ::llvm::APInt >(attr.getValue()) : (::std::nullopt);
}

} // namespace detail
PackOpAdaptor::PackOpAdaptor(PackOp op) : PackOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PackOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_batched_lvls;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PackOp::getBatchedLvlsAttrName(*odsOpName)) {
      tblgen_batched_lvls = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_batched_lvls && !(((tblgen_batched_lvls.isa<::mlir::IntegerAttr>())) && ((tblgen_batched_lvls.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.pack' op ""attribute 'batched_lvls' failed to satisfy constraint: index attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PackOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PackOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PackOp::getValues() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> PackOp::getCoordinates() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange PackOp::getValuesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PackOp::getCoordinatesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PackOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PackOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PackOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr PackOp::getBatchedLvlsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBatchedLvlsAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional< ::llvm::APInt > PackOp::getBatchedLvls() {
  auto attr = getBatchedLvlsAttr();
  return attr ? ::std::optional< ::llvm::APInt >(attr.getValue()) : (::std::nullopt);
}

void PackOp::setBatchedLvlsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getBatchedLvlsAttrName(), attr);
}

void PackOp::setBatchedLvls(::std::optional<::llvm::APInt> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getBatchedLvlsAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), *attrValue));
    (*this)->removeAttr(getBatchedLvlsAttrName());
}

::mlir::Attribute PackOp::removeBatchedLvlsAttr() {
  return (*this)->removeAttr(getBatchedLvlsAttrName());
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value values, ::mlir::Value coordinates, /*optional*/::mlir::IntegerAttr batched_lvls) {
  odsState.addOperands(values);
  odsState.addOperands(coordinates);
  if (batched_lvls) {
    odsState.addAttribute(getBatchedLvlsAttrName(odsState.name), batched_lvls);
  }
  odsState.addTypes(result);
}

void PackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values, ::mlir::Value coordinates, /*optional*/::mlir::IntegerAttr batched_lvls) {
  odsState.addOperands(values);
  odsState.addOperands(coordinates);
  if (batched_lvls) {
    odsState.addAttribute(getBatchedLvlsAttrName(odsState.name), batched_lvls);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PackOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_batched_lvls;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBatchedLvlsAttrName()) {
      tblgen_batched_lvls = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps5(*this, tblgen_batched_lvls, "batched_lvls")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PackOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult PackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valuesRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valuesOperands(valuesRawOperands);  ::llvm::SMLoc valuesOperandsLoc;
  (void)valuesOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand coordinatesRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> coordinatesOperands(coordinatesRawOperands);  ::llvm::SMLoc coordinatesOperandsLoc;
  (void)coordinatesOperandsLoc;
  ::mlir::IntegerAttr batched_lvlsAttr;
  ::mlir::Type valuesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valuesTypes(valuesRawTypes);
  ::mlir::Type coordinatesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> coordinatesTypes(coordinatesRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  valuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valuesRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  coordinatesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(coordinatesRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("batched_lvls"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(batched_lvlsAttr, parser.getBuilder().getIndexType(), "batched_lvls",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valuesRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    coordinatesRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valuesOperands, valuesTypes, valuesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coordinatesOperands, coordinatesTypes, coordinatesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PackOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValues();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCoordinates();
  if ((*this)->getAttr("batched_lvls")) {
    _odsPrinter << ' ' << "batched_lvls";
    _odsPrinter << ' ' << "=";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getBatchedLvlsAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("batched_lvls");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValues().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getCoordinates().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void PackOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::PackOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::PushBackOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PushBackOpGenericAdaptorBase::PushBackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.push_back", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> PushBackOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr PushBackOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr PushBackOpGenericAdaptorBase::getInboundsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PushBackOp::getInboundsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool PushBackOpGenericAdaptorBase::getInbounds() {
  auto attr = getInboundsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
PushBackOpAdaptor::PushBackOpAdaptor(PushBackOp op) : PushBackOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult PushBackOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inbounds;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PushBackOp::getInboundsAttrName(*odsOpName)) {
      tblgen_inbounds = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_inbounds && !((tblgen_inbounds.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'sparse_tensor.push_back' op ""attribute 'inbounds' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

void PushBackOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outBuffer");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "newSize");
}

std::pair<unsigned, unsigned> PushBackOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range PushBackOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> PushBackOp::getCurSize() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> PushBackOp::getInBuffer() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::Value PushBackOp::getValue() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
}

::mlir::TypedValue<::mlir::IndexType> PushBackOp::getN() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::TypedValue<::mlir::IndexType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*operands.begin());
}

::mlir::MutableOperandRange PushBackOp::getCurSizeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PushBackOp::getInBufferMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PushBackOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PushBackOp::getNMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PushBackOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PushBackOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> PushBackOp::getOutBuffer() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSResults(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> PushBackOp::getNewSize() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(1).begin());
}

::mlir::UnitAttr PushBackOp::getInboundsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getInboundsAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool PushBackOp::getInbounds() {
  auto attr = getInboundsAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void PushBackOp::setInboundsAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInboundsAttrName(), attr);
}

void PushBackOp::setInbounds(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInboundsAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInboundsAttrName());
}

::mlir::Attribute PushBackOp::removeInboundsAttr() {
  return (*this)->removeAttr(getInboundsAttrName());
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outBuffer, ::mlir::Type newSize, ::mlir::Value curSize, ::mlir::Value inBuffer, ::mlir::Value value, /*optional*/::mlir::Value n, /*optional*/::mlir::UnitAttr inbounds) {
  odsState.addOperands(curSize);
  odsState.addOperands(inBuffer);
  odsState.addOperands(value);
  if (n)
    odsState.addOperands(n);
  if (inbounds) {
    odsState.addAttribute(getInboundsAttrName(odsState.name), inbounds);
  }
  odsState.addTypes(outBuffer);
  odsState.addTypes(newSize);
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value curSize, ::mlir::Value inBuffer, ::mlir::Value value, /*optional*/::mlir::Value n, /*optional*/::mlir::UnitAttr inbounds) {
  odsState.addOperands(curSize);
  odsState.addOperands(inBuffer);
  odsState.addOperands(value);
  if (n)
    odsState.addOperands(n);
  if (inbounds) {
    odsState.addAttribute(getInboundsAttrName(odsState.name), inbounds);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(PushBackOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value curSize, ::mlir::Value inBuffer, ::mlir::Value value, /*optional*/::mlir::Value n, /*optional*/::mlir::UnitAttr inbounds) {
  odsState.addOperands(curSize);
  odsState.addOperands(inBuffer);
  odsState.addOperands(value);
  if (n)
    odsState.addOperands(n);
  if (inbounds) {
    odsState.addAttribute(getInboundsAttrName(odsState.name), inbounds);
  }
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outBuffer, ::mlir::Type newSize, ::mlir::Value curSize, ::mlir::Value inBuffer, ::mlir::Value value, /*optional*/::mlir::Value n, /*optional*/bool inbounds) {
  odsState.addOperands(curSize);
  odsState.addOperands(inBuffer);
  odsState.addOperands(value);
  if (n)
    odsState.addOperands(n);
  if (inbounds) {
    odsState.addAttribute(getInboundsAttrName(odsState.name), ((inbounds) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addTypes(outBuffer);
  odsState.addTypes(newSize);
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value curSize, ::mlir::Value inBuffer, ::mlir::Value value, /*optional*/::mlir::Value n, /*optional*/bool inbounds) {
  odsState.addOperands(curSize);
  odsState.addOperands(inBuffer);
  odsState.addOperands(value);
  if (n)
    odsState.addOperands(n);
  if (inbounds) {
    odsState.addAttribute(getInboundsAttrName(odsState.name), ((inbounds) ? odsBuilder.getUnitAttr() : nullptr));
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(PushBackOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value curSize, ::mlir::Value inBuffer, ::mlir::Value value, /*optional*/::mlir::Value n, /*optional*/bool inbounds) {
  odsState.addOperands(curSize);
  odsState.addOperands(inBuffer);
  odsState.addOperands(value);
  if (n)
    odsState.addOperands(n);
  if (inbounds) {
    odsState.addAttribute(getInboundsAttrName(odsState.name), ((inbounds) ? odsBuilder.getUnitAttr() : nullptr));
  }
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PushBackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PushBackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(PushBackOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 2u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult PushBackOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_inbounds;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getInboundsAttrName()) {
      tblgen_inbounds = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps0(*this, tblgen_inbounds, "inbounds")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<ShapedType>().getElementType(), (*this->getODSOperands(2).begin()).getType()))))
    return emitOpError("failed to verify that value type matches element type of inBuffer");
  if (!((((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()))))
    return emitOpError("failed to verify that all of {inBuffer, outBuffer} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult PushBackOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult PushBackOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(2);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[1].getType();
  ::mlir::Type odsInferredType1 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  inferredReturnTypes[1] = odsInferredType1;
  return ::mlir::success();
}

::mlir::ParseResult PushBackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand curSizeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> curSizeOperands(curSizeRawOperands);  ::llvm::SMLoc curSizeOperandsLoc;
  (void)curSizeOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand inBufferRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inBufferOperands(inBufferRawOperands);  ::llvm::SMLoc inBufferOperandsLoc;
  (void)inBufferOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> nOperands;
  ::llvm::SMLoc nOperandsLoc;
  (void)nOperandsLoc;
  ::mlir::Type curSizeRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> curSizeTypes(curSizeRawTypes);
  ::mlir::Type inBufferRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inBufferTypes(inBufferRawTypes);
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> nTypes;
  if (::mlir::succeeded(parser.parseOptionalKeyword("inbounds"))) {
    result.addAttribute("inbounds", parser.getBuilder().getUnitAttr());
  }

  curSizeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(curSizeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  inBufferOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inBufferRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    nOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      nOperands.push_back(operand);
    }
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IndexType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    curSizeRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inBufferRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      nTypes.push_back(optionalType);
    }
  }
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(inBufferTypes[0]);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(curSizeOperands, curSizeTypes, curSizeOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inBufferOperands, inBufferTypes, inBufferOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(nOperands, nTypes, nOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PushBackOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("inbounds")) {
    _odsPrinter << ' ' << "inbounds";
  }
  _odsPrinter << ' ';
  _odsPrinter << getCurSize();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getInBuffer();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  if (getN()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getN())
      _odsPrinter << value;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("inbounds");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInboundsAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inbounds");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getCurSize().getType();
    if (auto validType = type.dyn_cast<::mlir::IndexType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getInBuffer().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (getN()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << (getN() ? ::llvm::ArrayRef<::mlir::Type>(getN().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::PushBackOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ReduceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReduceOpGenericAdaptorBase::ReduceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.reduce", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReduceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReduceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &ReduceOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ReduceOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ReduceOpAdaptor::ReduceOpAdaptor(ReduceOp op) : ReduceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ReduceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReduceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReduceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceOp::getX() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::Value ReduceOp::getY() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(1).begin());
}

::mlir::Value ReduceOp::getIdentity() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(2).begin());
}

::mlir::MutableOperandRange ReduceOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReduceOp::getYMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReduceOp::getIdentityMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReduceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReduceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReduceOp::getOutput() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Region &ReduceOp::getRegion() {
  return (*this)->getRegion(0);
}

void ReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value x, ::mlir::Value y, ::mlir::Value identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  odsState.addOperands(identity);
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void ReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value y, ::mlir::Value identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  odsState.addOperands(identity);
  (void)odsState.addRegion();

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ReduceOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value y, ::mlir::Value identity) {
  odsState.addOperands(x);
  odsState.addOperands(y);
  odsState.addOperands(identity);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReduceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ReduceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ReduceOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ReduceOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReduceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult ReduceOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ReduceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand yRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> yOperands(yRawOperands);  ::llvm::SMLoc yOperandsLoc;
  (void)yOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand identityRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> identityOperands(identityRawOperands);  ::llvm::SMLoc identityOperandsLoc;
  (void)identityOperandsLoc;
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  yOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(yRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  identityOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(identityRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawTypes[0] = type;
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addTypes(outputTypes);
  if (parser.resolveOperands(xOperands, outputTypes[0], xOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(yOperands, outputTypes[0], yOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(identityOperands, outputTypes[0], identityOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReduceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getX();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getY();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getIdentity();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
}

void ReduceOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ReduceOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::SelectOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SelectOpGenericAdaptorBase::SelectOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.select", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SelectOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SelectOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &SelectOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange SelectOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
SelectOpAdaptor::SelectOpAdaptor(SelectOp op) : SelectOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SelectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::getX() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SelectOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::getOutput() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Region &SelectOp::getRegion() {
  return (*this)->getRegion(0);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value x) {
  odsState.addOperands(x);
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x) {
  odsState.addOperands(x);
  (void)odsState.addRegion();

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x) {
  odsState.addOperands(x);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SelectOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SelectOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult SelectOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addTypes(xTypes[0]);
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SelectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getX();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getRegion());
}

void SelectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SelectOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::SetStorageSpecifierOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SetStorageSpecifierOpGenericAdaptorBase::SetStorageSpecifierOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.storage_specifier.set", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SetStorageSpecifierOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SetStorageSpecifierOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::sparse_tensor::StorageSpecifierKindAttr SetStorageSpecifierOpGenericAdaptorBase::getSpecifierKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SetStorageSpecifierOp::getSpecifierKindAttrName(*odsOpName)).cast<::mlir::sparse_tensor::StorageSpecifierKindAttr>();
  return attr;
}

::mlir::sparse_tensor::StorageSpecifierKind SetStorageSpecifierOpGenericAdaptorBase::getSpecifierKind() {
  auto attr = getSpecifierKindAttr();
  return attr.getValue();
}

::mlir::IntegerAttr SetStorageSpecifierOpGenericAdaptorBase::getLevelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SetStorageSpecifierOp::getLevelAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<::mlir::sparse_tensor::Level> SetStorageSpecifierOpGenericAdaptorBase::getLevel() {
  auto attr = getLevelAttr();
  return attr ? ::std::optional<::mlir::sparse_tensor::Level>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
SetStorageSpecifierOpAdaptor::SetStorageSpecifierOpAdaptor(SetStorageSpecifierOp op) : SetStorageSpecifierOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SetStorageSpecifierOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_specifierKind;
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.storage_specifier.set' op ""requires attribute 'specifierKind'");
    if (namedAttrIt->getName() == SetStorageSpecifierOp::getSpecifierKindAttrName(*odsOpName)) {
      tblgen_specifierKind = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SetStorageSpecifierOp::getLevelAttrName(*odsOpName)) {
      tblgen_level = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_specifierKind && !((tblgen_specifierKind.isa<::mlir::sparse_tensor::StorageSpecifierKindAttr>())))
    return emitError(loc, "'sparse_tensor.storage_specifier.set' op ""attribute 'specifierKind' failed to satisfy constraint: sparse tensor storage specifier kind");

  if (tblgen_level && !(((tblgen_level.isa<::mlir::IntegerAttr>())) && ((tblgen_level.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.storage_specifier.set' op ""attribute 'level' failed to satisfy constraint: level attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SetStorageSpecifierOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SetStorageSpecifierOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType> SetStorageSpecifierOp::getSpecifier() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::IndexType> SetStorageSpecifierOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange SetStorageSpecifierOp::getSpecifierMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SetStorageSpecifierOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SetStorageSpecifierOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SetStorageSpecifierOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType> SetStorageSpecifierOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType>>(*getODSResults(0).begin());
}

::mlir::sparse_tensor::StorageSpecifierKindAttr SetStorageSpecifierOp::getSpecifierKindAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSpecifierKindAttrName()).cast<::mlir::sparse_tensor::StorageSpecifierKindAttr>();
}

::mlir::sparse_tensor::StorageSpecifierKind SetStorageSpecifierOp::getSpecifierKind() {
  auto attr = getSpecifierKindAttr();
  return attr.getValue();
}

::mlir::IntegerAttr SetStorageSpecifierOp::getLevelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getLevelAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<::mlir::sparse_tensor::Level> SetStorageSpecifierOp::getLevel() {
  auto attr = getLevelAttr();
  return attr ? ::std::optional<::mlir::sparse_tensor::Level>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void SetStorageSpecifierOp::setSpecifierKindAttr(::mlir::sparse_tensor::StorageSpecifierKindAttr attr) {
  (*this)->setAttr(getSpecifierKindAttrName(), attr);
}

void SetStorageSpecifierOp::setSpecifierKind(::mlir::sparse_tensor::StorageSpecifierKind attrValue) {
  (*this)->setAttr(getSpecifierKindAttrName(), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void SetStorageSpecifierOp::setLevelAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLevelAttrName(), attr);
}

void SetStorageSpecifierOp::setLevel(::std::optional<::mlir::sparse_tensor::Level> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getLevelAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), *attrValue));
    (*this)->removeAttr(getLevelAttrName());
}

::mlir::Attribute SetStorageSpecifierOp::removeLevelAttr() {
  return (*this)->removeAttr(getLevelAttrName());
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKind, /*optional*/::mlir::IntegerAttr level, ::mlir::Value value) {
  odsState.addOperands(specifier);
  odsState.addOperands(value);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), specifierKind);
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  odsState.addTypes(result);
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKind, /*optional*/::mlir::IntegerAttr level, ::mlir::Value value) {
  odsState.addOperands(specifier);
  odsState.addOperands(value);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), specifierKind);
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SetStorageSpecifierOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKind, /*optional*/::mlir::IntegerAttr level, ::mlir::Value value) {
  odsState.addOperands(specifier);
  odsState.addOperands(value);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), specifierKind);
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKind specifierKind, /*optional*/::mlir::IntegerAttr level, ::mlir::Value value) {
  odsState.addOperands(specifier);
  odsState.addOperands(value);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(odsBuilder.getContext(), specifierKind));
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  odsState.addTypes(result);
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKind specifierKind, /*optional*/::mlir::IntegerAttr level, ::mlir::Value value) {
  odsState.addOperands(specifier);
  odsState.addOperands(value);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(odsBuilder.getContext(), specifierKind));
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SetStorageSpecifierOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value specifier, ::mlir::sparse_tensor::StorageSpecifierKind specifierKind, /*optional*/::mlir::IntegerAttr level, ::mlir::Value value) {
  odsState.addOperands(specifier);
  odsState.addOperands(value);
  odsState.addAttribute(getSpecifierKindAttrName(odsState.name), ::mlir::sparse_tensor::StorageSpecifierKindAttr::get(odsBuilder.getContext(), specifierKind));
  if (level) {
    odsState.addAttribute(getLevelAttrName(odsState.name), level);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SetStorageSpecifierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SetStorageSpecifierOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SetStorageSpecifierOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_specifierKind;
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'specifierKind'");
    if (namedAttrIt->getName() == getSpecifierKindAttrName()) {
      tblgen_specifierKind = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getLevelAttrName()) {
      tblgen_level = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps3(*this, tblgen_specifierKind, "specifierKind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps4(*this, tblgen_level, "level")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()) && ((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {result, specifier} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult SetStorageSpecifierOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult SetStorageSpecifierOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SetStorageSpecifierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand specifierRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> specifierOperands(specifierRawOperands);  ::llvm::SMLoc specifierOperandsLoc;
  (void)specifierOperandsLoc;
  ::mlir::sparse_tensor::StorageSpecifierKindAttr specifierKindAttr;
  ::mlir::IntegerAttr levelAttr;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  specifierOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(specifierRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(specifierKindAttr, ::mlir::Type{}, "specifierKind",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("at"))) {

  if (parser.parseCustomAttributeWithFallback(levelAttr, parser.getBuilder().getIndexType(), "level",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseKeyword("with"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(specifierOperands, resultTypes[0], specifierOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SetStorageSpecifierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSpecifier();
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getSpecifierKindAttr());
  if ((*this)->getAttr("level")) {
    _odsPrinter << ' ' << "at";
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getLevelAttr());
  }
  _odsPrinter << ' ' << "with";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("specifierKind");
  elidedAttrs.push_back("level");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void SetStorageSpecifierOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SetStorageSpecifierOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::SortCooOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SortCooOpGenericAdaptorBase::SortCooOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.sort_coo", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SortCooOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr SortCooOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr SortCooOpGenericAdaptorBase::getNxAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SortCooOp::getNxAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional< ::llvm::APInt > SortCooOpGenericAdaptorBase::getNx() {
  auto attr = getNxAttr();
  return attr ? ::std::optional< ::llvm::APInt >(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr SortCooOpGenericAdaptorBase::getNyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SortCooOp::getNyAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional< ::llvm::APInt > SortCooOpGenericAdaptorBase::getNy() {
  auto attr = getNyAttr();
  return attr ? ::std::optional< ::llvm::APInt >(attr.getValue()) : (::std::nullopt);
}

::mlir::sparse_tensor::SparseTensorSortKindAttr SortCooOpGenericAdaptorBase::getAlgorithmAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SortCooOp::getAlgorithmAttrName(*odsOpName)).cast<::mlir::sparse_tensor::SparseTensorSortKindAttr>();
  return attr;
}

::mlir::sparse_tensor::SparseTensorSortKind SortCooOpGenericAdaptorBase::getAlgorithm() {
  auto attr = getAlgorithmAttr();
  return attr.getValue();
}

} // namespace detail
SortCooOpAdaptor::SortCooOpAdaptor(SortCooOp op) : SortCooOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SortCooOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_algorithm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.sort_coo' op ""requires attribute 'algorithm'");
    if (namedAttrIt->getName() == SortCooOp::getAlgorithmAttrName(*odsOpName)) {
      tblgen_algorithm = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_nx;
  ::mlir::Attribute tblgen_ny;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SortCooOp::getNxAttrName(*odsOpName)) {
      tblgen_nx = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SortCooOp::getNyAttrName(*odsOpName)) {
      tblgen_ny = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_nx && !(((tblgen_nx.isa<::mlir::IntegerAttr>())) && ((tblgen_nx.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.sort_coo' op ""attribute 'nx' failed to satisfy constraint: index attribute");

  if (tblgen_ny && !(((tblgen_ny.isa<::mlir::IntegerAttr>())) && ((tblgen_ny.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.sort_coo' op ""attribute 'ny' failed to satisfy constraint: index attribute");

  if (tblgen_algorithm && !((tblgen_algorithm.isa<::mlir::sparse_tensor::SparseTensorSortKindAttr>())))
    return emitError(loc, "'sparse_tensor.sort_coo' op ""attribute 'algorithm' failed to satisfy constraint: sparse tensor sort algorithm");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SortCooOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range SortCooOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> SortCooOp::getN() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::MemRefType> SortCooOp::getXy() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::MemRefType>>(*getODSOperands(1).begin());
}

::mlir::Operation::operand_range SortCooOp::getYs() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SortCooOp::getNMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SortCooOp::getXyMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SortCooOp::getYsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SortCooOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SortCooOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr SortCooOp::getNxAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getNxAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional< ::llvm::APInt > SortCooOp::getNx() {
  auto attr = getNxAttr();
  return attr ? ::std::optional< ::llvm::APInt >(attr.getValue()) : (::std::nullopt);
}

::mlir::IntegerAttr SortCooOp::getNyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getNyAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional< ::llvm::APInt > SortCooOp::getNy() {
  auto attr = getNyAttr();
  return attr ? ::std::optional< ::llvm::APInt >(attr.getValue()) : (::std::nullopt);
}

::mlir::sparse_tensor::SparseTensorSortKindAttr SortCooOp::getAlgorithmAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlgorithmAttrName()).cast<::mlir::sparse_tensor::SparseTensorSortKindAttr>();
}

::mlir::sparse_tensor::SparseTensorSortKind SortCooOp::getAlgorithm() {
  auto attr = getAlgorithmAttr();
  return attr.getValue();
}

void SortCooOp::setNxAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNxAttrName(), attr);
}

void SortCooOp::setNx(::std::optional<::llvm::APInt> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNxAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), *attrValue));
    (*this)->removeAttr(getNxAttrName());
}

void SortCooOp::setNyAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNyAttrName(), attr);
}

void SortCooOp::setNy(::std::optional<::llvm::APInt> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getNyAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), *attrValue));
    (*this)->removeAttr(getNyAttrName());
}

void SortCooOp::setAlgorithmAttr(::mlir::sparse_tensor::SparseTensorSortKindAttr attr) {
  (*this)->setAttr(getAlgorithmAttrName(), attr);
}

void SortCooOp::setAlgorithm(::mlir::sparse_tensor::SparseTensorSortKind attrValue) {
  (*this)->setAttr(getAlgorithmAttrName(), ::mlir::sparse_tensor::SparseTensorSortKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

::mlir::Attribute SortCooOp::removeNxAttr() {
  return (*this)->removeAttr(getNxAttrName());
}

::mlir::Attribute SortCooOp::removeNyAttr() {
  return (*this)->removeAttr(getNyAttrName());
}

void SortCooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value n, ::mlir::Value xy, ::mlir::ValueRange ys, /*optional*/::mlir::IntegerAttr nx, /*optional*/::mlir::IntegerAttr ny, ::mlir::sparse_tensor::SparseTensorSortKindAttr algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xy);
  odsState.addOperands(ys);
  if (nx) {
    odsState.addAttribute(getNxAttrName(odsState.name), nx);
  }
  if (ny) {
    odsState.addAttribute(getNyAttrName(odsState.name), ny);
  }
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), algorithm);
}

void SortCooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value n, ::mlir::Value xy, ::mlir::ValueRange ys, /*optional*/::mlir::IntegerAttr nx, /*optional*/::mlir::IntegerAttr ny, ::mlir::sparse_tensor::SparseTensorSortKindAttr algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xy);
  odsState.addOperands(ys);
  if (nx) {
    odsState.addAttribute(getNxAttrName(odsState.name), nx);
  }
  if (ny) {
    odsState.addAttribute(getNyAttrName(odsState.name), ny);
  }
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), algorithm);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SortCooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value n, ::mlir::Value xy, ::mlir::ValueRange ys, /*optional*/::mlir::IntegerAttr nx, /*optional*/::mlir::IntegerAttr ny, ::mlir::sparse_tensor::SparseTensorSortKind algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xy);
  odsState.addOperands(ys);
  if (nx) {
    odsState.addAttribute(getNxAttrName(odsState.name), nx);
  }
  if (ny) {
    odsState.addAttribute(getNyAttrName(odsState.name), ny);
  }
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), ::mlir::sparse_tensor::SparseTensorSortKindAttr::get(odsBuilder.getContext(), algorithm));
}

void SortCooOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value n, ::mlir::Value xy, ::mlir::ValueRange ys, /*optional*/::mlir::IntegerAttr nx, /*optional*/::mlir::IntegerAttr ny, ::mlir::sparse_tensor::SparseTensorSortKind algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xy);
  odsState.addOperands(ys);
  if (nx) {
    odsState.addAttribute(getNxAttrName(odsState.name), nx);
  }
  if (ny) {
    odsState.addAttribute(getNyAttrName(odsState.name), ny);
  }
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), ::mlir::sparse_tensor::SparseTensorSortKindAttr::get(odsBuilder.getContext(), algorithm));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SortCooOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SortCooOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_algorithm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'algorithm'");
    if (namedAttrIt->getName() == getAlgorithmAttrName()) {
      tblgen_algorithm = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_nx;
  ::mlir::Attribute tblgen_ny;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNxAttrName()) {
      tblgen_nx = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNyAttrName()) {
      tblgen_ny = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps5(*this, tblgen_nx, "nx")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps5(*this, tblgen_ny, "ny")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps6(*this, tblgen_algorithm, "algorithm")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SortCooOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SortCooOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::sparse_tensor::SparseTensorSortKindAttr algorithmAttr;
  ::mlir::OpAsmParser::UnresolvedOperand nRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> nOperands(nRawOperands);  ::llvm::SMLoc nOperandsLoc;
  (void)nOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand xyRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xyOperands(xyRawOperands);  ::llvm::SMLoc xyOperandsLoc;
  (void)xyOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ysOperands;
  ::llvm::SMLoc ysOperandsLoc;
  (void)ysOperandsLoc;
  ::mlir::Type xyRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xyTypes(xyRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> ysTypes;

  if (parser.parseCustomAttributeWithFallback(algorithmAttr, ::mlir::Type{}, "algorithm",
          result.attributes)) {
    return ::mlir::failure();
  }

  nOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(nRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  xyOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xyRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("jointly"))) {

  ysOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ysOperands))
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
    xyRawTypes[0] = type;
  }
  if (::mlir::succeeded(parser.parseOptionalKeyword("jointly"))) {

  if (parser.parseTypeList(ysTypes))
    return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(nOperands, odsBuildableType0, nOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(xyOperands, xyTypes, xyOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ysOperands, ysTypes, ysOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SortCooOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getAlgorithmAttr());
  _odsPrinter << ' ';
  _odsPrinter << getN();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getXy();
  if (!getYs().empty()) {
    _odsPrinter << ' ' << "jointly";
    _odsPrinter << ' ';
    _odsPrinter << getYs();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("algorithm");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getXy().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getYs().empty()) {
    _odsPrinter << ' ' << "jointly";
    _odsPrinter << ' ';
    _odsPrinter << getYs().getTypes();
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SortCooOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::SortOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SortOpGenericAdaptorBase::SortOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.sort", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SortOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SortOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr SortOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::sparse_tensor::SparseTensorSortKindAttr SortOpGenericAdaptorBase::getAlgorithmAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SortOp::getAlgorithmAttrName(*odsOpName)).cast<::mlir::sparse_tensor::SparseTensorSortKindAttr>();
  return attr;
}

::mlir::sparse_tensor::SparseTensorSortKind SortOpGenericAdaptorBase::getAlgorithm() {
  auto attr = getAlgorithmAttr();
  return attr.getValue();
}

} // namespace detail
SortOpAdaptor::SortOpAdaptor(SortOp op) : SortOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult SortOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_algorithm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.sort' op ""requires attribute 'algorithm'");
    if (namedAttrIt->getName() == SortOp::getAlgorithmAttrName(*odsOpName)) {
      tblgen_algorithm = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.sort' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SortOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'sparse_tensor.sort' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_algorithm && !((tblgen_algorithm.isa<::mlir::sparse_tensor::SparseTensorSortKindAttr>())))
    return emitError(loc, "'sparse_tensor.sort' op ""attribute 'algorithm' failed to satisfy constraint: sparse tensor sort algorithm");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SortOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range SortOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> SortOp::getN() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range SortOp::getXs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SortOp::getYs() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SortOp::getNMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SortOp::getXsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SortOp::getYsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SortOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SortOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::sparse_tensor::SparseTensorSortKindAttr SortOp::getAlgorithmAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAlgorithmAttrName()).cast<::mlir::sparse_tensor::SparseTensorSortKindAttr>();
}

::mlir::sparse_tensor::SparseTensorSortKind SortOp::getAlgorithm() {
  auto attr = getAlgorithmAttr();
  return attr.getValue();
}

void SortOp::setAlgorithmAttr(::mlir::sparse_tensor::SparseTensorSortKindAttr attr) {
  (*this)->setAttr(getAlgorithmAttrName(), attr);
}

void SortOp::setAlgorithm(::mlir::sparse_tensor::SparseTensorSortKind attrValue) {
  (*this)->setAttr(getAlgorithmAttrName(), ::mlir::sparse_tensor::SparseTensorSortKindAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void SortOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value n, ::mlir::ValueRange xs, ::mlir::ValueRange ys, ::mlir::sparse_tensor::SparseTensorSortKindAttr algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xs);
  odsState.addOperands(ys);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(xs.size()), static_cast<int32_t>(ys.size())}));
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), algorithm);
}

void SortOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value n, ::mlir::ValueRange xs, ::mlir::ValueRange ys, ::mlir::sparse_tensor::SparseTensorSortKindAttr algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xs);
  odsState.addOperands(ys);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(xs.size()), static_cast<int32_t>(ys.size())}));
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), algorithm);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SortOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value n, ::mlir::ValueRange xs, ::mlir::ValueRange ys, ::mlir::sparse_tensor::SparseTensorSortKind algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xs);
  odsState.addOperands(ys);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(xs.size()), static_cast<int32_t>(ys.size())}));
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), ::mlir::sparse_tensor::SparseTensorSortKindAttr::get(odsBuilder.getContext(), algorithm));
}

void SortOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value n, ::mlir::ValueRange xs, ::mlir::ValueRange ys, ::mlir::sparse_tensor::SparseTensorSortKind algorithm) {
  odsState.addOperands(n);
  odsState.addOperands(xs);
  odsState.addOperands(ys);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(xs.size()), static_cast<int32_t>(ys.size())}));
  odsState.addAttribute(getAlgorithmAttrName(odsState.name), ::mlir::sparse_tensor::SparseTensorSortKindAttr::get(odsBuilder.getContext(), algorithm));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SortOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SortOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_algorithm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'algorithm'");
    if (namedAttrIt->getName() == getAlgorithmAttrName()) {
      tblgen_algorithm = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps6(*this, tblgen_algorithm, "algorithm")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SortOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SortOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::sparse_tensor::SparseTensorSortKindAttr algorithmAttr;
  ::mlir::OpAsmParser::UnresolvedOperand nRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> nOperands(nRawOperands);  ::llvm::SMLoc nOperandsLoc;
  (void)nOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> xsOperands;
  ::llvm::SMLoc xsOperandsLoc;
  (void)xsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ysOperands;
  ::llvm::SMLoc ysOperandsLoc;
  (void)ysOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> xsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> ysTypes;

  if (parser.parseCustomAttributeWithFallback(algorithmAttr, ::mlir::Type{}, "algorithm",
          result.attributes)) {
    return ::mlir::failure();
  }

  nOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(nRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  xsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(xsOperands))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("jointly"))) {

  ysOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ysOperands))
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(xsTypes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("jointly"))) {

  if (parser.parseTypeList(ysTypes))
    return ::mlir::failure();
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(xsOperands.size()), static_cast<int32_t>(ysOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(nOperands, odsBuildableType0, nOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(xsOperands, xsTypes, xsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ysOperands, ysTypes, ysOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SortOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getAlgorithmAttr());
  _odsPrinter << ' ';
  _odsPrinter << getN();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getXs();
  if (!getYs().empty()) {
    _odsPrinter << ' ' << "jointly";
    _odsPrinter << ' ';
    _odsPrinter << getYs();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("algorithm");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getXs().getTypes();
  if (!getYs().empty()) {
    _odsPrinter << ' ' << "jointly";
    _odsPrinter << ' ';
    _odsPrinter << getYs().getTypes();
  }
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::SortOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::StorageSpecifierInitOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
StorageSpecifierInitOpGenericAdaptorBase::StorageSpecifierInitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.storage_specifier.init", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> StorageSpecifierInitOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr StorageSpecifierInitOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
StorageSpecifierInitOpAdaptor::StorageSpecifierInitOpAdaptor(StorageSpecifierInitOp op) : StorageSpecifierInitOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult StorageSpecifierInitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StorageSpecifierInitOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range StorageSpecifierInitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType> StorageSpecifierInitOp::getSource() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType>{} : ::llvm::cast<::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType>>(*operands.begin());
}

::mlir::MutableOperandRange StorageSpecifierInitOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StorageSpecifierInitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StorageSpecifierInitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType> StorageSpecifierInitOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::sparse_tensor::StorageSpecifierType>>(*getODSResults(0).begin());
}

void StorageSpecifierInitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type result) {
      build(odsBuilder, odsState, result, Value());
    
}

void StorageSpecifierInitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Value source) {
  if (source)
    odsState.addOperands(source);
  odsState.addTypes(result);
}

void StorageSpecifierInitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value source) {
  if (source)
    odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StorageSpecifierInitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StorageSpecifierInitOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StorageSpecifierInitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult StorageSpecifierInitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sourceOperands;
  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> sourceTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("with"))) {

  {
    sourceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sourceOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("from"))) {

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sourceTypes.push_back(optionalType);
    }
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();
  }

  if (parser.parseType(resultRawTypes[0]))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StorageSpecifierInitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (getSource()) {
    _odsPrinter << ' ' << "with";
    _odsPrinter << ' ';
    if (::mlir::Value value = getSource())
      _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  if (getSource()) {
    _odsPrinter << ' ' << "from";
    _odsPrinter << ' ';
    _odsPrinter << (getSource() ? ::llvm::ArrayRef<::mlir::Type>(getSource().getType()) : ::llvm::ArrayRef<::mlir::Type>());
    _odsPrinter << ' ' << "to";
  }
  _odsPrinter << ' ';
   _odsPrinter << getResult().getType();
}

void StorageSpecifierInitOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::StorageSpecifierInitOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ToCoordinatesBufferOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToCoordinatesBufferOpGenericAdaptorBase::ToCoordinatesBufferOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.coordinates_buffer", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ToCoordinatesBufferOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ToCoordinatesBufferOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ToCoordinatesBufferOpAdaptor::ToCoordinatesBufferOpAdaptor(ToCoordinatesBufferOp op) : ToCoordinatesBufferOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ToCoordinatesBufferOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ToCoordinatesBufferOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ToCoordinatesBufferOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ToCoordinatesBufferOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ToCoordinatesBufferOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ToCoordinatesBufferOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ToCoordinatesBufferOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ToCoordinatesBufferOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void ToCoordinatesBufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(result);
}

void ToCoordinatesBufferOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToCoordinatesBufferOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ToCoordinatesBufferOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ToCoordinatesBufferOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ToCoordinatesBufferOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToCoordinatesBufferOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToCoordinatesBufferOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ToCoordinatesBufferOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ToCoordinatesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToCoordinatesOpGenericAdaptorBase::ToCoordinatesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.coordinates", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ToCoordinatesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ToCoordinatesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ToCoordinatesOpGenericAdaptorBase::getLevelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ToCoordinatesOp::getLevelAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::sparse_tensor::Level ToCoordinatesOpGenericAdaptorBase::getLevel() {
  auto attr = getLevelAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ToCoordinatesOpAdaptor::ToCoordinatesOpAdaptor(ToCoordinatesOp op) : ToCoordinatesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ToCoordinatesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.coordinates' op ""requires attribute 'level'");
    if (namedAttrIt->getName() == ToCoordinatesOp::getLevelAttrName(*odsOpName)) {
      tblgen_level = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_level && !(((tblgen_level.isa<::mlir::IntegerAttr>())) && ((tblgen_level.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.coordinates' op ""attribute 'level' failed to satisfy constraint: level attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ToCoordinatesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ToCoordinatesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ToCoordinatesOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ToCoordinatesOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ToCoordinatesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ToCoordinatesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ToCoordinatesOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::IntegerAttr ToCoordinatesOp::getLevelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLevelAttrName()).cast<::mlir::IntegerAttr>();
}

::mlir::sparse_tensor::Level ToCoordinatesOp::getLevel() {
  auto attr = getLevelAttr();
  return attr.getValue().getZExtValue();
}

void ToCoordinatesOp::setLevelAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLevelAttrName(), attr);
}

void ToCoordinatesOp::setLevel(::mlir::sparse_tensor::Level attrValue) {
  (*this)->setAttr(getLevelAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), attrValue));
}

void ToCoordinatesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::IntegerAttr level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), level);
  odsState.addTypes(result);
}

void ToCoordinatesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::IntegerAttr level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), level);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToCoordinatesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::sparse_tensor::Level level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), level));
  odsState.addTypes(result);
}

void ToCoordinatesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::sparse_tensor::Level level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), level));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToCoordinatesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ToCoordinatesOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'level'");
    if (namedAttrIt->getName() == getLevelAttrName()) {
      tblgen_level = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps4(*this, tblgen_level, "level")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ToCoordinatesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ToCoordinatesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToCoordinatesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToCoordinatesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ToCoordinatesOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ToPositionsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToPositionsOpGenericAdaptorBase::ToPositionsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.positions", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ToPositionsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ToPositionsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ToPositionsOpGenericAdaptorBase::getLevelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ToPositionsOp::getLevelAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

::mlir::sparse_tensor::Level ToPositionsOpGenericAdaptorBase::getLevel() {
  auto attr = getLevelAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ToPositionsOpAdaptor::ToPositionsOpAdaptor(ToPositionsOp op) : ToPositionsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ToPositionsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.positions' op ""requires attribute 'level'");
    if (namedAttrIt->getName() == ToPositionsOp::getLevelAttrName(*odsOpName)) {
      tblgen_level = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_level && !(((tblgen_level.isa<::mlir::IntegerAttr>())) && ((tblgen_level.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.positions' op ""attribute 'level' failed to satisfy constraint: level attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ToPositionsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ToPositionsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ToPositionsOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ToPositionsOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ToPositionsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ToPositionsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ToPositionsOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::IntegerAttr ToPositionsOp::getLevelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getLevelAttrName()).cast<::mlir::IntegerAttr>();
}

::mlir::sparse_tensor::Level ToPositionsOp::getLevel() {
  auto attr = getLevelAttr();
  return attr.getValue().getZExtValue();
}

void ToPositionsOp::setLevelAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLevelAttrName(), attr);
}

void ToPositionsOp::setLevel(::mlir::sparse_tensor::Level attrValue) {
  (*this)->setAttr(getLevelAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), attrValue));
}

void ToPositionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::IntegerAttr level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), level);
  odsState.addTypes(result);
}

void ToPositionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::IntegerAttr level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), level);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToPositionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::sparse_tensor::Level level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), level));
  odsState.addTypes(result);
}

void ToPositionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::sparse_tensor::Level level) {
  odsState.addOperands(tensor);
  odsState.addAttribute(getLevelAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), level));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToPositionsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ToPositionsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_level;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'level'");
    if (namedAttrIt->getName() == getLevelAttrName()) {
      tblgen_level = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps4(*this, tblgen_level, "level")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ToPositionsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ToPositionsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToPositionsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToPositionsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ToPositionsOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ToSliceOffsetOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToSliceOffsetOpGenericAdaptorBase::ToSliceOffsetOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.slice.offset", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ToSliceOffsetOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ToSliceOffsetOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ToSliceOffsetOpGenericAdaptorBase::getDimAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ToSliceOffsetOp::getDimAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

::llvm::APInt ToSliceOffsetOpGenericAdaptorBase::getDim() {
  auto attr = getDimAttr();
  return attr.getValue();
}

} // namespace detail
ToSliceOffsetOpAdaptor::ToSliceOffsetOpAdaptor(ToSliceOffsetOp op) : ToSliceOffsetOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ToSliceOffsetOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dim;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.slice.offset' op ""requires attribute 'dim'");
    if (namedAttrIt->getName() == ToSliceOffsetOp::getDimAttrName(*odsOpName)) {
      tblgen_dim = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dim && !(((tblgen_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.slice.offset' op ""attribute 'dim' failed to satisfy constraint: index attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ToSliceOffsetOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ToSliceOffsetOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ToSliceOffsetOp::getSlice() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ToSliceOffsetOp::getSliceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ToSliceOffsetOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ToSliceOffsetOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> ToSliceOffsetOp::getOffset() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr ToSliceOffsetOp::getDimAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDimAttrName()).cast<::mlir::IntegerAttr>();
}

::llvm::APInt ToSliceOffsetOp::getDim() {
  auto attr = getDimAttr();
  return attr.getValue();
}

void ToSliceOffsetOp::setDimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDimAttrName(), attr);
}

void ToSliceOffsetOp::setDim(::llvm::APInt attrValue) {
  (*this)->setAttr(getDimAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), attrValue));
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type offset, ::mlir::Value slice, ::mlir::IntegerAttr dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), dim);
  odsState.addTypes(offset);
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value slice, ::mlir::IntegerAttr dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), dim);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ToSliceOffsetOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value slice, ::mlir::IntegerAttr dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), dim);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type offset, ::mlir::Value slice, ::llvm::APInt dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dim));
  odsState.addTypes(offset);
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value slice, ::llvm::APInt dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dim));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ToSliceOffsetOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value slice, ::llvm::APInt dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dim));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ToSliceOffsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ToSliceOffsetOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ToSliceOffsetOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dim;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dim'");
    if (namedAttrIt->getName() == getDimAttrName()) {
      tblgen_dim = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps5(*this, tblgen_dim, "dim")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ToSliceOffsetOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult ToSliceOffsetOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ToSliceOffsetOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sliceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sliceOperands(sliceRawOperands);  ::llvm::SMLoc sliceOperandsLoc;
  (void)sliceOperandsLoc;
  ::mlir::IntegerAttr dimAttr;
  ::mlir::Type sliceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sliceTypes(sliceRawTypes);

  sliceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sliceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("at"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(dimAttr, parser.getBuilder().getIndexType(), "dim",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sliceRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(sliceOperands, sliceTypes, sliceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToSliceOffsetOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSlice();
  _odsPrinter << ' ' << "at";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getDimAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("dim");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSlice().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToSliceOffsetOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ToSliceOffsetOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ToSliceStrideOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToSliceStrideOpGenericAdaptorBase::ToSliceStrideOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.slice.stride", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ToSliceStrideOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ToSliceStrideOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ToSliceStrideOpGenericAdaptorBase::getDimAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ToSliceStrideOp::getDimAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

::llvm::APInt ToSliceStrideOpGenericAdaptorBase::getDim() {
  auto attr = getDimAttr();
  return attr.getValue();
}

} // namespace detail
ToSliceStrideOpAdaptor::ToSliceStrideOpAdaptor(ToSliceStrideOp op) : ToSliceStrideOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ToSliceStrideOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dim;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'sparse_tensor.slice.stride' op ""requires attribute 'dim'");
    if (namedAttrIt->getName() == ToSliceStrideOp::getDimAttrName(*odsOpName)) {
      tblgen_dim = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dim && !(((tblgen_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_dim.cast<::mlir::IntegerAttr>().getType().isa<::mlir::IndexType>()))))
    return emitError(loc, "'sparse_tensor.slice.stride' op ""attribute 'dim' failed to satisfy constraint: index attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ToSliceStrideOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ToSliceStrideOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ToSliceStrideOp::getSlice() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ToSliceStrideOp::getSliceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ToSliceStrideOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ToSliceStrideOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::IndexType> ToSliceStrideOp::getStride() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::IndexType>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr ToSliceStrideOp::getDimAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getDimAttrName()).cast<::mlir::IntegerAttr>();
}

::llvm::APInt ToSliceStrideOp::getDim() {
  auto attr = getDimAttr();
  return attr.getValue();
}

void ToSliceStrideOp::setDimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDimAttrName(), attr);
}

void ToSliceStrideOp::setDim(::llvm::APInt attrValue) {
  (*this)->setAttr(getDimAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIndexType(), attrValue));
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type stride, ::mlir::Value slice, ::mlir::IntegerAttr dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), dim);
  odsState.addTypes(stride);
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value slice, ::mlir::IntegerAttr dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), dim);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ToSliceStrideOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value slice, ::mlir::IntegerAttr dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), dim);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type stride, ::mlir::Value slice, ::llvm::APInt dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dim));
  odsState.addTypes(stride);
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value slice, ::llvm::APInt dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dim));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ToSliceStrideOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value slice, ::llvm::APInt dim) {
  odsState.addOperands(slice);
  odsState.addAttribute(getDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIndexType(), dim));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToSliceStrideOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ToSliceStrideOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ToSliceStrideOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ToSliceStrideOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dim;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dim'");
    if (namedAttrIt->getName() == getDimAttrName()) {
      tblgen_dim = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SparseTensorOps5(*this, tblgen_dim, "dim")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ToSliceStrideOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult ToSliceStrideOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getIndexType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ToSliceStrideOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sliceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sliceOperands(sliceRawOperands);  ::llvm::SMLoc sliceOperandsLoc;
  (void)sliceOperandsLoc;
  ::mlir::IntegerAttr dimAttr;
  ::mlir::Type sliceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sliceTypes(sliceRawTypes);

  sliceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sliceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("at"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(dimAttr, parser.getBuilder().getIndexType(), "dim",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sliceRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(sliceOperands, sliceTypes, sliceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToSliceStrideOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSlice();
  _odsPrinter << ' ' << "at";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getDimAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("dim");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSlice().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToSliceStrideOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ToSliceStrideOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::ToValuesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ToValuesOpGenericAdaptorBase::ToValuesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.values", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ToValuesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ToValuesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ToValuesOpAdaptor::ToValuesOpAdaptor(ToValuesOp op) : ToValuesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult ToValuesOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ToValuesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ToValuesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ToValuesOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ToValuesOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ToValuesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ToValuesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ToValuesOp::getResult() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

void ToValuesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(result);
}

void ToValuesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToValuesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ToValuesOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ToValuesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ToValuesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ToValuesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ToValuesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::ToValuesOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::UnaryOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UnaryOpGenericAdaptorBase::UnaryOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.unary", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UnaryOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr UnaryOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &UnaryOpGenericAdaptorBase::getPresentRegion() {
  return *odsRegions[0];
}

::mlir::Region &UnaryOpGenericAdaptorBase::getAbsentRegion() {
  return *odsRegions[1];
}

::mlir::RegionRange UnaryOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
UnaryOpAdaptor::UnaryOpAdaptor(UnaryOp op) : UnaryOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UnaryOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UnaryOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnaryOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnaryOp::getX() {
  return ::llvm::cast<::mlir::Value>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange UnaryOp::getXMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UnaryOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnaryOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UnaryOp::getOutput() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(0).begin());
}

::mlir::Region &UnaryOp::getPresentRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &UnaryOp::getAbsentRegion() {
  return (*this)->getRegion(1);
}

void UnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value x) {
  odsState.addOperands(x);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(output);
}

void UnaryOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x) {
  odsState.addOperands(x);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnaryOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UnaryOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps0(*this, region, "presentRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_SparseTensorOps0(*this, region, "absentRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult UnaryOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult UnaryOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  ::mlir::Type outputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> outputTypes(outputRawTypes);
  std::unique_ptr<::mlir::Region> presentRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> absentRegionRegion = std::make_unique<::mlir::Region>();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    outputRawTypes[0] = type;
  }
  if (parser.parseKeyword("present"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseRegion(*presentRegionRegion))
    return ::mlir::failure();
  if (parser.parseKeyword("absent"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseRegion(*absentRegionRegion))
    return ::mlir::failure();
  result.addRegion(std::move(presentRegionRegion));
  result.addRegion(std::move(absentRegionRegion));
  result.addTypes(outputTypes);
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UnaryOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getX();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getX().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getOutput().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printNewline();
  _odsPrinter << ' ' << "present";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getPresentRegion());
  _odsPrinter.printNewline();
  _odsPrinter << ' ' << "absent";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getAbsentRegion());
}

void UnaryOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::UnaryOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::UnpackOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
UnpackOpGenericAdaptorBase::UnpackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.unpack", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> UnpackOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr UnpackOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
UnpackOpAdaptor::UnpackOpAdaptor(UnpackOp op) : UnpackOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult UnpackOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void UnpackOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "values");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "coordinates");
  auto resultGroup2 = getODSResults(2);
  if (!resultGroup2.empty())
    setNameFn(*resultGroup2.begin(), "nse");
}

std::pair<unsigned, unsigned> UnpackOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnpackOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> UnpackOp::getTensor() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange UnpackOp::getTensorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UnpackOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnpackOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::RankedTensorType> UnpackOp::getValues() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(0).begin());
}

::mlir::TypedValue<::mlir::RankedTensorType> UnpackOp::getCoordinates() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::RankedTensorType>>(*getODSResults(1).begin());
}

::mlir::Value UnpackOp::getNse() {
  return ::llvm::cast<::mlir::Value>(*getODSResults(2).begin());
}

void UnpackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type values, ::mlir::Type coordinates, ::mlir::Type nse, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  odsState.addTypes(values);
  odsState.addTypes(coordinates);
  odsState.addTypes(nse);
}

void UnpackOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor) {
  odsState.addOperands(tensor);
  assert(resultTypes.size() == 3u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnpackOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 3u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UnpackOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSResults(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UnpackOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult UnpackOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(tensorRawOperands);  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::mlir::Type tensorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tensorTypes(tensorRawTypes);
  ::mlir::Type valuesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valuesTypes(valuesRawTypes);
  ::mlir::Type coordinatesRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> coordinatesTypes(coordinatesRawTypes);
  ::mlir::Type nseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> nseTypes(nseRawTypes);

  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tensorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valuesRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::RankedTensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    coordinatesRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    nseRawTypes[0] = type;
  }
  result.addTypes(valuesTypes);
  result.addTypes(coordinatesTypes);
  result.addTypes(nseTypes);
  if (parser.resolveOperands(tensorOperands, tensorTypes, tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UnpackOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTensor();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTensor().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getValues().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getCoordinates().getType();
    if (auto validType = type.dyn_cast<::mlir::RankedTensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getNse().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UnpackOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::UnpackOp)

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
// ::mlir::sparse_tensor::YieldOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
YieldOpGenericAdaptorBase::YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("sparse_tensor.yield", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> YieldOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
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

::mlir::DictionaryAttr YieldOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : YieldOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

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

::mlir::Value YieldOp::getResult() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value{} : ::llvm::cast<::mlir::Value>(*operands.begin());
}

::mlir::MutableOperandRange YieldOp::getResultMutable() {
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
      build(odsBuilder, odsState, Value());
    
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value result) {
  if (result)
    odsState.addOperands(result);
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value result) {
  if (result)
    odsState.addOperands(result);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
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

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SparseTensorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> resultOperands;
  ::llvm::SMLoc resultOperandsLoc;
  (void)resultOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> resultTypes;

  {
    resultOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      resultOperands.push_back(operand);
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      resultTypes.push_back(optionalType);
    }
  }
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  if (::mlir::Value value = getResult())
    _odsPrinter << value;
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getResult() ? ::llvm::ArrayRef<::mlir::Type>(getResult().getType()) : ::llvm::ArrayRef<::mlir::Type>());
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace sparse_tensor
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::sparse_tensor::YieldOp)


#endif  // GET_OP_CLASSES

