/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::vector::VectorScaleOp,
::mlir::vector::BitCastOp,
::mlir::vector::BroadcastOp,
::mlir::vector::CompressStoreOp,
::mlir::vector::ConstantMaskOp,
::mlir::vector::ContractionOp,
::mlir::vector::CreateMaskOp,
::mlir::vector::ExpandLoadOp,
::mlir::vector::ExtractElementOp,
::mlir::vector::ExtractMapOp,
::mlir::vector::ExtractOp,
::mlir::vector::ExtractStridedSliceOp,
::mlir::vector::FMAOp,
::mlir::vector::FlatTransposeOp,
::mlir::vector::GatherOp,
::mlir::vector::InsertElementOp,
::mlir::vector::InsertMapOp,
::mlir::vector::InsertOp,
::mlir::vector::InsertStridedSliceOp,
::mlir::vector::LoadOp,
::mlir::vector::MaskedLoadOp,
::mlir::vector::MaskedStoreOp,
::mlir::vector::MatmulOp,
::mlir::vector::MultiDimReductionOp,
::mlir::vector::OuterProductOp,
::mlir::vector::PrintOp,
::mlir::vector::ReductionOp,
::mlir::vector::ReshapeOp,
::mlir::vector::ScanOp,
::mlir::vector::ScatterOp,
::mlir::vector::ShapeCastOp,
::mlir::vector::ShuffleOp,
::mlir::vector::SplatOp,
::mlir::vector::StoreOp,
::mlir::vector::TransferReadOp,
::mlir::vector::TransferWriteOp,
::mlir::vector::TransposeOp,
::mlir::vector::TypeCastOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace vector {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IndexType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::VectorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getRank()
                           == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 1-bit signless integer values of ranks 1, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getRank()
                           == 1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be  of ranks 1, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::VectorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessIntOrIndex()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be signless integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isSignedInteger())) || ((elementType.isa<::mlir::IndexType>())) || ((elementType.isa<::mlir::FloatType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getRank()
                           == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of signless integer or signed integer or index or floating-point values of ranks 1, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::IntegerType>())) || ((elementType.isa<::mlir::IndexType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getRank()
                           == 1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be vector of integer or index values of ranks 1, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger())) || ((type.isa<::mlir::IndexType>())) || ((type.isa<::mlir::FloatType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer/index/float type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::ShapedType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be shaped of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ((type.cast<::mlir::ShapedType>().hasRank()))) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be ranked tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_VectorOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((type.cast<::mlir::ShapedType>().hasStaticShape())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be statically shaped memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::vector::CombiningKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Kind of combining function for contractions and reductions";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::AffineMapAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: AffineMap attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_VectorOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::BoolAttr>()); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 1-bit boolean array attribute";
  }
  return ::mlir::success();
}
} // namespace vector
} // namespace mlir
namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::VectorScaleOp definitions
//===----------------------------------------------------------------------===//

VectorScaleOpAdaptor::VectorScaleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

VectorScaleOpAdaptor::VectorScaleOpAdaptor(VectorScaleOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange VectorScaleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VectorScaleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VectorScaleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr VectorScaleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VectorScaleOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VectorScaleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VectorScaleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> VectorScaleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VectorScaleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VectorScaleOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::Value VectorScaleOp::res() {
  return *getODSResults(0).begin();
}

void VectorScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void VectorScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(VectorScaleOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void VectorScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VectorScaleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void VectorScaleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(VectorScaleOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult VectorScaleOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VectorScaleOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult VectorScaleOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = odsBuilder.getIndexType();
  return ::mlir::success();
}

::mlir::ParseResult VectorScaleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void VectorScaleOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void VectorScaleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::VectorScaleOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::BitCastOp definitions
//===----------------------------------------------------------------------===//

BitCastOpAdaptor::BitCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

BitCastOpAdaptor::BitCastOpAdaptor(BitCastOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange BitCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitCastOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitCastOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BitCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitCastOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value BitCastOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitCastOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BitCastOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitCastOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value BitCastOp::result() {
  return *getODSResults(0).begin();
}

void BitCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(result);
}

void BitCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType().cast<::mlir::ShapedType>().getRank(), (*this->getODSResults(0).begin()).getType().cast<::mlir::ShapedType>().getRank()})))))
    return emitOpError("failed to verify that all of {source, result} have same rank");
  return ::mlir::success();
}

::mlir::LogicalResult BitCastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BitCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
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
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BitCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BitCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::BitCastOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::BroadcastOp definitions
//===----------------------------------------------------------------------===//

BroadcastOpAdaptor::BroadcastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

BroadcastOpAdaptor::BroadcastOpAdaptor(BroadcastOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange BroadcastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BroadcastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BroadcastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BroadcastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BroadcastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BroadcastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BroadcastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value BroadcastOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BroadcastOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange BroadcastOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BroadcastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BroadcastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BroadcastOp::getVector() {
  return *getODSResults(0).begin();
}

::mlir::Value BroadcastOp::vector() {
  return *getODSResults(0).begin();
}

void BroadcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(vector);
}

void BroadcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BroadcastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that source operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult BroadcastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult BroadcastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);

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
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  result.addTypes(vectorTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BroadcastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BroadcastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::BroadcastOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::CompressStoreOp definitions
//===----------------------------------------------------------------------===//

CompressStoreOpAdaptor::CompressStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CompressStoreOpAdaptor::CompressStoreOpAdaptor(CompressStoreOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CompressStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CompressStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange CompressStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CompressStoreOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value CompressStoreOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CompressStoreOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange CompressStoreOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value CompressStoreOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value CompressStoreOpAdaptor::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value CompressStoreOpAdaptor::getValueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::Value CompressStoreOpAdaptor::valueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr CompressStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CompressStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CompressStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
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

::mlir::Operation::operand_range CompressStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CompressStoreOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value CompressStoreOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CompressStoreOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CompressStoreOp::indices() {
  return getODSOperands(1);
}

::mlir::Value CompressStoreOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value CompressStoreOp::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value CompressStoreOp::getValueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::Value CompressStoreOp::valueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange CompressStoreOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::getValueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CompressStoreOp::valueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CompressStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CompressStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CompressStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(valueToStore);
}

void CompressStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(valueToStore);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CompressStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CompressStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CompressStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CompressStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType valueToStoreRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueToStoreOperands(valueToStoreRawOperands);  ::llvm::SMLoc valueToStoreOperandsLoc;
  (void)valueToStoreOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type valueToStoreRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueToStoreTypes(valueToStoreRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valueToStoreOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueToStoreRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueToStoreRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueToStoreOperands, valueToStoreTypes, valueToStoreOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CompressStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getValueToStore();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getValueToStore().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void CompressStoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::CompressStoreOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ConstantMaskOp definitions
//===----------------------------------------------------------------------===//

ConstantMaskOpAdaptor::ConstantMaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ConstantMaskOpAdaptor::ConstantMaskOpAdaptor(ConstantMaskOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ConstantMaskOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstantMaskOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantMaskOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ConstantMaskOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ConstantMaskOpAdaptor::getMaskDimSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("mask_dim_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ConstantMaskOpAdaptor::getMaskDimSizes() {
  auto attr = getMaskDimSizesAttr();
  return attr;
}

::mlir::ArrayAttr ConstantMaskOpAdaptor::mask_dim_sizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("mask_dim_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ConstantMaskOpAdaptor::mask_dim_sizes() {
  auto attr = mask_dim_sizesAttr();
  return attr;
}

::mlir::LogicalResult ConstantMaskOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_mask_dim_sizes = odsAttrs.get("mask_dim_sizes");
    if (!tblgen_mask_dim_sizes)
      return emitError(loc, "'vector.constant_mask' op ""requires attribute 'mask_dim_sizes'");

    if (tblgen_mask_dim_sizes && !(((tblgen_mask_dim_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_mask_dim_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.constant_mask' op ""attribute 'mask_dim_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantMaskOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantMaskOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantMaskOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantMaskOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ConstantMaskOp::getMaskDimSizesAttr() {
  return (*this)->getAttr(getMaskDimSizesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ConstantMaskOp::getMaskDimSizes() {
  auto attr = getMaskDimSizesAttr();
  return attr;
}

::mlir::ArrayAttr ConstantMaskOp::mask_dim_sizesAttr() {
  return (*this)->getAttr(mask_dim_sizesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ConstantMaskOp::mask_dim_sizes() {
  auto attr = mask_dim_sizesAttr();
  return attr;
}

void ConstantMaskOp::setMaskDimSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMaskDimSizesAttrName(), attr);
}

void ConstantMaskOp::mask_dim_sizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(mask_dim_sizesAttrName(), attr);
}

void ConstantMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ArrayAttr mask_dim_sizes) {
  odsState.addAttribute(getMaskDimSizesAttrName(odsState.name), mask_dim_sizes);
  odsState.addTypes(resultType0);
}

void ConstantMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr mask_dim_sizes) {
  odsState.addAttribute(getMaskDimSizesAttrName(odsState.name), mask_dim_sizes);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantMaskOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantMaskOp::verifyInvariantsImpl() {
  {
    auto tblgen_mask_dim_sizes = (*this)->getAttr(getMaskDimSizesAttrName());
    if (!tblgen_mask_dim_sizes)
      return emitOpError("requires attribute 'mask_dim_sizes'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_mask_dim_sizes, "mask_dim_sizes")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantMaskOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ConstantMaskOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr mask_dim_sizesAttr;
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(mask_dim_sizesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "mask_dim_sizes",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  return ::mlir::success();
}

void ConstantMaskOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMaskDimSizesAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"mask_dim_sizes"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ConstantMaskOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ConstantMaskOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ContractionOp definitions
//===----------------------------------------------------------------------===//

ContractionOpAdaptor::ContractionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ContractionOpAdaptor::ContractionOpAdaptor(ContractionOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ContractionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ContractionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ContractionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ContractionOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ContractionOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ContractionOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value ContractionOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value ContractionOpAdaptor::getAcc() {
  return *getODSOperands(2).begin();
}

::mlir::Value ContractionOpAdaptor::acc() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange ContractionOpAdaptor::getMasks() {
  return getODSOperands(3);
}

::mlir::ValueRange ContractionOpAdaptor::masks() {
  return getODSOperands(3);
}

::mlir::DictionaryAttr ContractionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ContractionOpAdaptor::getIndexingMapsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::llvm::SmallVector<::mlir::AffineMap, 4> ContractionOpAdaptor::getIndexingMaps() {
  auto attr = getIndexingMapsAttr();
  return llvm::to_vector<4>(attr.getAsValueRange<::mlir::AffineMapAttr>());;
}

::mlir::ArrayAttr ContractionOpAdaptor::indexing_mapsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("indexing_maps").cast<::mlir::ArrayAttr>();
  return attr;
}

::llvm::SmallVector<::mlir::AffineMap, 4> ContractionOpAdaptor::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return llvm::to_vector<4>(attr.getAsValueRange<::mlir::AffineMapAttr>());;
}

::mlir::ArrayAttr ContractionOpAdaptor::getIteratorTypesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ContractionOpAdaptor::getIteratorTypes() {
  auto attr = getIteratorTypesAttr();
  return attr;
}

::mlir::ArrayAttr ContractionOpAdaptor::iterator_typesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("iterator_types").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ContractionOpAdaptor::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::vector::CombiningKindAttr ContractionOpAdaptor::getKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
  if (!attr)
    attr = ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext());
  return attr;
}

::mlir::vector::CombiningKind ContractionOpAdaptor::getKind() {
  auto attr = getKindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext()).getKind();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr ContractionOpAdaptor::kindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
  if (!attr)
    attr = ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext());
  return attr;
}

::mlir::vector::CombiningKind ContractionOpAdaptor::kind() {
  auto attr = kindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext()).getKind();
  return attr.getKind();
}

::mlir::LogicalResult ContractionOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_indexing_maps = odsAttrs.get("indexing_maps");
    if (!tblgen_indexing_maps)
      return emitError(loc, "'vector.contract' op ""requires attribute 'indexing_maps'");

    if (tblgen_indexing_maps && !(((tblgen_indexing_maps.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_indexing_maps.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::AffineMapAttr>()); }))))
      return emitError(loc, "'vector.contract' op ""attribute 'indexing_maps' failed to satisfy constraint: AffineMap array attribute");
  }
  {
    auto tblgen_iterator_types = odsAttrs.get("iterator_types");
    if (!tblgen_iterator_types)
      return emitError(loc, "'vector.contract' op ""requires attribute 'iterator_types'");

    if (tblgen_iterator_types && !((tblgen_iterator_types.isa<::mlir::ArrayAttr>())))
      return emitError(loc, "'vector.contract' op ""attribute 'iterator_types' failed to satisfy constraint: array attribute");
  }
  {
    auto tblgen_kind = odsAttrs.get("kind");
    if (tblgen_kind && !((tblgen_kind.isa<::mlir::vector::CombiningKindAttr>())))
      return emitError(loc, "'vector.contract' op ""attribute 'kind' failed to satisfy constraint: Kind of combining function for contractions and reductions");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ContractionOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ContractionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ContractionOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ContractionOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ContractionOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value ContractionOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value ContractionOp::getAcc() {
  return *getODSOperands(2).begin();
}

::mlir::Value ContractionOp::acc() {
  return *getODSOperands(2).begin();
}

::mlir::Operation::operand_range ContractionOp::getMasks() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ContractionOp::masks() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange ContractionOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::getAccMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::accMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::getMasksMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ContractionOp::masksMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ContractionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ContractionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ContractionOp::getIndexingMapsAttr() {
  return (*this)->getAttr(getIndexingMapsAttrName()).cast<::mlir::ArrayAttr>();
}

::llvm::SmallVector<::mlir::AffineMap, 4> ContractionOp::getIndexingMaps() {
  auto attr = getIndexingMapsAttr();
  return llvm::to_vector<4>(attr.getAsValueRange<::mlir::AffineMapAttr>());;
}

::mlir::ArrayAttr ContractionOp::indexing_mapsAttr() {
  return (*this)->getAttr(indexing_mapsAttrName()).cast<::mlir::ArrayAttr>();
}

::llvm::SmallVector<::mlir::AffineMap, 4> ContractionOp::indexing_maps() {
  auto attr = indexing_mapsAttr();
  return llvm::to_vector<4>(attr.getAsValueRange<::mlir::AffineMapAttr>());;
}

::mlir::ArrayAttr ContractionOp::getIteratorTypesAttr() {
  return (*this)->getAttr(getIteratorTypesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ContractionOp::getIteratorTypes() {
  auto attr = getIteratorTypesAttr();
  return attr;
}

::mlir::ArrayAttr ContractionOp::iterator_typesAttr() {
  return (*this)->getAttr(iterator_typesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ContractionOp::iterator_types() {
  auto attr = iterator_typesAttr();
  return attr;
}

::mlir::vector::CombiningKindAttr ContractionOp::getKindAttr() {
  return (*this)->getAttr(getKindAttrName()).dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind ContractionOp::getKind() {
  auto attr = getKindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder((*this)->getContext()).getContext()).getKind();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr ContractionOp::kindAttr() {
  return (*this)->getAttr(kindAttrName()).dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind ContractionOp::kind() {
  auto attr = kindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder((*this)->getContext()).getContext()).getKind();
  return attr.getKind();
}

void ContractionOp::setIndexingMapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIndexingMapsAttrName(), attr);
}

void ContractionOp::indexing_mapsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(indexing_mapsAttrName(), attr);
}

void ContractionOp::setIteratorTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIteratorTypesAttrName(), attr);
}

void ContractionOp::iterator_typesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(iterator_typesAttrName(), attr);
}

void ContractionOp::setKindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(getKindAttrName(), attr);
}

void ContractionOp::kindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(kindAttrName(), attr);
}

void ContractionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKindAttr kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addOperands(masks);
  odsState.addAttribute(getIndexingMapsAttrName(odsState.name), indexing_maps);
  odsState.addAttribute(getIteratorTypesAttrName(odsState.name), iterator_types);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addTypes(resultType0);
}

void ContractionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKindAttr kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addOperands(masks);
  odsState.addAttribute(getIndexingMapsAttrName(odsState.name), indexing_maps);
  odsState.addAttribute(getIteratorTypesAttrName(odsState.name), iterator_types);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ContractionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::llvm::SmallVector<::mlir::AffineMap, 4> indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKind kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addOperands(masks);
  odsState.addAttribute(getIndexingMapsAttrName(odsState.name), odsBuilder.getAffineMapArrayAttr(indexing_maps));
  odsState.addAttribute(getIteratorTypesAttrName(odsState.name), iterator_types);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addTypes(resultType0);
}

void ContractionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc, ::mlir::ValueRange masks, ::llvm::SmallVector<::mlir::AffineMap, 4> indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::vector::CombiningKind kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addOperands(masks);
  odsState.addAttribute(getIndexingMapsAttrName(odsState.name), odsBuilder.getAffineMapArrayAttr(indexing_maps));
  odsState.addAttribute(getIteratorTypesAttrName(odsState.name), iterator_types);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ContractionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ContractionOp::verifyInvariantsImpl() {
  {
    auto tblgen_indexing_maps = (*this)->getAttr(getIndexingMapsAttrName());
    if (!tblgen_indexing_maps)
      return emitOpError("requires attribute 'indexing_maps'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps1(*this, tblgen_indexing_maps, "indexing_maps")))
      return ::mlir::failure();
  }
  {
    auto tblgen_iterator_types = (*this)->getAttr(getIteratorTypesAttrName());
    if (!tblgen_iterator_types)
      return emitOpError("requires attribute 'iterator_types'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps2(*this, tblgen_iterator_types, "iterator_types")))
      return ::mlir::failure();
  }
  {
    auto tblgen_kind = (*this)->getAttr(getKindAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps3(*this, tblgen_kind, "kind")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getOperation()).getNumOperands() > 1)) && (((*this->getOperation()).getOperand(0).getType().isa<::mlir::ShapedType>())) && (((*this->getOperation()).getOperand(1).getType().isa<::mlir::ShapedType>())) && ((::mlir::getElementTypeOrSelf((*this->getOperation()).getOperand(0)) == ::mlir::getElementTypeOrSelf((*this->getOperation()).getOperand(1))))))
    return emitOpError("failed to verify that lhs and rhs have same element type");
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(2)))))
    return emitOpError("failed to verify that third operand acc and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult ContractionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ContractionOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ContractionOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::CreateMaskOp definitions
//===----------------------------------------------------------------------===//

CreateMaskOpAdaptor::CreateMaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CreateMaskOpAdaptor::CreateMaskOpAdaptor(CreateMaskOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange CreateMaskOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CreateMaskOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CreateMaskOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange CreateMaskOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr CreateMaskOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CreateMaskOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateMaskOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CreateMaskOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateMaskOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CreateMaskOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CreateMaskOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateMaskOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CreateMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addTypes(resultType0);
}

void CreateMaskOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateMaskOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateMaskOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateMaskOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateMaskOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << operands();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void CreateMaskOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::CreateMaskOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExpandLoadOp definitions
//===----------------------------------------------------------------------===//

ExpandLoadOpAdaptor::ExpandLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ExpandLoadOpAdaptor::ExpandLoadOpAdaptor(ExpandLoadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ExpandLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExpandLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ExpandLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpandLoadOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExpandLoadOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExpandLoadOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange ExpandLoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value ExpandLoadOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value ExpandLoadOpAdaptor::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value ExpandLoadOpAdaptor::getPassThru() {
  return *getODSOperands(3).begin();
}

::mlir::Value ExpandLoadOpAdaptor::pass_thru() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr ExpandLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExpandLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExpandLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
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

::mlir::Operation::operand_range ExpandLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpandLoadOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExpandLoadOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ExpandLoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ExpandLoadOp::indices() {
  return getODSOperands(1);
}

::mlir::Value ExpandLoadOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value ExpandLoadOp::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value ExpandLoadOp::getPassThru() {
  return *getODSOperands(3).begin();
}

::mlir::Value ExpandLoadOp::pass_thru() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange ExpandLoadOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::getPassThruMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExpandLoadOp::pass_thruMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExpandLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpandLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpandLoadOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value ExpandLoadOp::result() {
  return *getODSResults(0).begin();
}

void ExpandLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addTypes(result);
}

void ExpandLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpandLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExpandLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExpandLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExpandLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType pass_thruRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> pass_thruOperands(pass_thruRawOperands);  ::llvm::SMLoc pass_thruOperandsLoc;
  (void)pass_thruOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type pass_thruRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> pass_thruTypes(pass_thruRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  pass_thruOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(pass_thruRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    pass_thruRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(pass_thruOperands, pass_thruTypes, pass_thruOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExpandLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPassThru();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getPassThru().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExpandLoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ExpandLoadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractElementOp definitions
//===----------------------------------------------------------------------===//

ExtractElementOpAdaptor::ExtractElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ExtractElementOpAdaptor::ExtractElementOpAdaptor(ExtractElementOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ExtractElementOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ExtractElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOpAdaptor::getPosition() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExtractElementOpAdaptor::position() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr ExtractElementOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExtractElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ExtractElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOp::getPosition() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ExtractElementOp::position() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange ExtractElementOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractElementOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractElementOp::getPositionMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractElementOp::positionMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value ExtractElementOp::result() {
  return *getODSResults(0).begin();
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, /*optional*/::mlir::Value position) {
  odsState.addOperands(vector);
  if (position)
    odsState.addOperands(position);
  odsState.addTypes(result);
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, /*optional*/::mlir::Value position) {
  odsState.addOperands(vector);
  if (position)
    odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(0).begin()).getType().cast<ShapedType>().getElementType(), (*this->getODSResults(0).begin()).getType()))))
    return emitOpError("failed to verify that result type matches element type of vector operand");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractElementOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExtractElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> positionOperands;
  ::llvm::SMLoc positionOperandsLoc;
  (void)positionOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> positionTypes;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  {
    positionOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      positionOperands.push_back(operand);
    }
  }
  if (!positionOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      positionTypes.push_back(optionalType);
    }
  }
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  for (::mlir::Type type : vectorTypes) {
    (void)type;
    if (!(((type.isa<::mlir::VectorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'vector' must be vector of any type values, but got " << type;
    }
  }
  result.addTypes(vectorTypes[0].cast<ShapedType>().getElementType());
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(positionOperands, positionTypes, positionOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractElementOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << "[";
  if (getPosition()) {
  if (::mlir::Value value = getPosition())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getPosition() ? ::llvm::ArrayRef<::mlir::Type>(getPosition().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExtractElementOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractElementOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractMapOp definitions
//===----------------------------------------------------------------------===//

ExtractMapOpAdaptor::ExtractMapOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ExtractMapOpAdaptor::ExtractMapOpAdaptor(ExtractMapOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ExtractMapOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractMapOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ExtractMapOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractMapOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractMapOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ExtractMapOpAdaptor::getIds() {
  return getODSOperands(1);
}

::mlir::ValueRange ExtractMapOpAdaptor::ids() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr ExtractMapOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExtractMapOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractMapOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ExtractMapOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractMapOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractMapOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ExtractMapOp::getIds() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ExtractMapOp::ids() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ExtractMapOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractMapOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractMapOp::getIdsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractMapOp::idsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractMapOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractMapOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ExtractMapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value vector, ::mlir::ValueRange ids) {
  odsState.addOperands(vector);
  odsState.addOperands(ids);
  odsState.addTypes(resultType0);
}

void ExtractMapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ValueRange ids) {
  odsState.addOperands(vector);
  odsState.addOperands(ids);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractMapOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractMapOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractMapOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExtractMapOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> idsOperands;
  ::llvm::SMLoc idsOperandsLoc;
  (void)idsOperandsLoc;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  idsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(idsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(idsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractMapOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << "[";
  _odsPrinter << getIds();
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ExtractMapOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractMapOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractOp definitions
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

::mlir::Value ExtractOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExtractOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExtractOpAdaptor::getPositionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("position").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractOpAdaptor::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

::mlir::ArrayAttr ExtractOpAdaptor::positionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("position").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractOpAdaptor::position() {
  auto attr = positionAttr();
  return attr;
}

::mlir::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_position = odsAttrs.get("position");
    if (!tblgen_position)
      return emitError(loc, "'vector.extract' op ""requires attribute 'position'");

    if (tblgen_position && !(((tblgen_position.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_position.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.extract' op ""attribute 'position' failed to satisfy constraint: 64-bit integer array attribute");
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

::mlir::Value ExtractOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExtractOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractOp::vectorMutable() {
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

::mlir::ArrayAttr ExtractOp::getPositionAttr() {
  return (*this)->getAttr(getPositionAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractOp::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

::mlir::ArrayAttr ExtractOp::positionAttr() {
  return (*this)->getAttr(positionAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractOp::position() {
  auto attr = positionAttr();
  return attr;
}

void ExtractOp::setPositionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPositionAttrName(), attr);
}

void ExtractOp::positionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(positionAttrName(), attr);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value vector, ::mlir::ArrayAttr position) {
  odsState.addOperands(vector);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  odsState.addTypes(resultType0);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value vector, ::mlir::ArrayAttr position) {
  odsState.addOperands(vector);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ExtractOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ArrayAttr position) {
  odsState.addOperands(vector);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
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

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ExtractOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ExtractOp::verifyInvariantsImpl() {
  {
    auto tblgen_position = (*this)->getAttr(getPositionAttrName());
    if (!tblgen_position)
      return emitOpError("requires attribute 'position'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_position, "position")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExtractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::ArrayAttr positionAttr;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(positionAttr, parser.getBuilder().getType<::mlir::NoneType>(), "position",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();

  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed(ExtractOp::inferReturnTypes(parser.getContext(),
      result.location, result.operands,
      result.attributes.getDictionary(parser.getContext()),
      result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
  return ::mlir::success();
}

void ExtractOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter.printAttributeWithoutType(getPositionAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"position"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ExtractOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ExtractStridedSliceOp definitions
//===----------------------------------------------------------------------===//

ExtractStridedSliceOpAdaptor::ExtractStridedSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ExtractStridedSliceOpAdaptor::ExtractStridedSliceOpAdaptor(ExtractStridedSliceOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ExtractStridedSliceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractStridedSliceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractStridedSliceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractStridedSliceOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractStridedSliceOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExtractStridedSliceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::getOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::offsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::offsets() {
  auto attr = offsetsAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::getSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::getSizes() {
  auto attr = getSizesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::sizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::sizes() {
  auto attr = sizesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOpAdaptor::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::LogicalResult ExtractStridedSliceOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_offsets = odsAttrs.get("offsets");
    if (!tblgen_offsets)
      return emitError(loc, "'vector.extract_strided_slice' op ""requires attribute 'offsets'");

    if (tblgen_offsets && !(((tblgen_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_offsets.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.extract_strided_slice' op ""attribute 'offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
    auto tblgen_sizes = odsAttrs.get("sizes");
    if (!tblgen_sizes)
      return emitError(loc, "'vector.extract_strided_slice' op ""requires attribute 'sizes'");

    if (tblgen_sizes && !(((tblgen_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.extract_strided_slice' op ""attribute 'sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
    auto tblgen_strides = odsAttrs.get("strides");
    if (!tblgen_strides)
      return emitError(loc, "'vector.extract_strided_slice' op ""requires attribute 'strides'");

    if (tblgen_strides && !(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.extract_strided_slice' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractStridedSliceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractStridedSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractStridedSliceOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractStridedSliceOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExtractStridedSliceOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractStridedSliceOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractStridedSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractStridedSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ExtractStridedSliceOp::getOffsetsAttr() {
  return (*this)->getAttr(getOffsetsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractStridedSliceOp::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOp::offsetsAttr() {
  return (*this)->getAttr(offsetsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractStridedSliceOp::offsets() {
  auto attr = offsetsAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOp::getSizesAttr() {
  return (*this)->getAttr(getSizesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractStridedSliceOp::getSizes() {
  auto attr = getSizesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOp::sizesAttr() {
  return (*this)->getAttr(sizesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractStridedSliceOp::sizes() {
  auto attr = sizesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOp::getStridesAttr() {
  return (*this)->getAttr(getStridesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractStridedSliceOp::getStrides() {
  auto attr = getStridesAttr();
  return attr;
}

::mlir::ArrayAttr ExtractStridedSliceOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractStridedSliceOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void ExtractStridedSliceOp::setOffsetsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOffsetsAttrName(), attr);
}

void ExtractStridedSliceOp::offsetsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(offsetsAttrName(), attr);
}

void ExtractStridedSliceOp::setSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getSizesAttrName(), attr);
}

void ExtractStridedSliceOp::sizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(sizesAttrName(), attr);
}

void ExtractStridedSliceOp::setStridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void ExtractStridedSliceOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void ExtractStridedSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value vector, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr sizes, ::mlir::ArrayAttr strides) {
  odsState.addOperands(vector);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
  odsState.addAttribute(getSizesAttrName(odsState.name), sizes);
  odsState.addAttribute(getStridesAttrName(odsState.name), strides);
  odsState.addTypes(resultType0);
}

void ExtractStridedSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr sizes, ::mlir::ArrayAttr strides) {
  odsState.addOperands(vector);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
  odsState.addAttribute(getSizesAttrName(odsState.name), sizes);
  odsState.addAttribute(getStridesAttrName(odsState.name), strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractStridedSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractStridedSliceOp::verifyInvariantsImpl() {
  {
    auto tblgen_offsets = (*this)->getAttr(getOffsetsAttrName());
    if (!tblgen_offsets)
      return emitOpError("requires attribute 'offsets'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_offsets, "offsets")))
      return ::mlir::failure();
  }
  {
    auto tblgen_sizes = (*this)->getAttr(getSizesAttrName());
    if (!tblgen_sizes)
      return emitOpError("requires attribute 'sizes'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_sizes, "sizes")))
      return ::mlir::failure();
  }
  {
    auto tblgen_strides = (*this)->getAttr(getStridesAttrName());
    if (!tblgen_strides)
      return emitOpError("requires attribute 'strides'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractStridedSliceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ExtractStridedSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);
  ::mlir::SmallVector<::mlir::Type, 1> allResultTypes;

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractStridedSliceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ExtractStridedSliceOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ExtractStridedSliceOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::FMAOp definitions
//===----------------------------------------------------------------------===//

FMAOpAdaptor::FMAOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FMAOpAdaptor::FMAOpAdaptor(FMAOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FMAOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMAOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMAOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOpAdaptor::getAcc() {
  return *getODSOperands(2).begin();
}

::mlir::Value FMAOpAdaptor::acc() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr FMAOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FMAOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FMAOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMAOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOp::getAcc() {
  return *getODSOperands(2).begin();
}

::mlir::Value FMAOp::acc() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FMAOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::getAccMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::accMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FMAOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMAOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value FMAOp::result() {
  return *getODSResults(0).begin();
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addTypes(result);
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMAOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value acc) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FMAOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FMAOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {lhs, rhs, acc, result} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult FMAOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FMAOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = operands[2].getType();
  return ::mlir::success();
}

::mlir::ParseResult FMAOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType accRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accOperands(accRawOperands);  ::llvm::SMLoc accOperandsLoc;
  (void)accOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  accOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  result.addTypes(lhsTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accOperands, lhsTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FMAOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getAcc();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FMAOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::FMAOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::FlatTransposeOp definitions
//===----------------------------------------------------------------------===//

FlatTransposeOpAdaptor::FlatTransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FlatTransposeOpAdaptor::FlatTransposeOpAdaptor(FlatTransposeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange FlatTransposeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FlatTransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FlatTransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FlatTransposeOpAdaptor::getMatrix() {
  return *getODSOperands(0).begin();
}

::mlir::Value FlatTransposeOpAdaptor::matrix() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FlatTransposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr FlatTransposeOpAdaptor::getRowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rows").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t FlatTransposeOpAdaptor::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr FlatTransposeOpAdaptor::rowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rows").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t FlatTransposeOpAdaptor::rows() {
  auto attr = rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr FlatTransposeOpAdaptor::getColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("columns").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t FlatTransposeOpAdaptor::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr FlatTransposeOpAdaptor::columnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("columns").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t FlatTransposeOpAdaptor::columns() {
  auto attr = columnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult FlatTransposeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_rows = odsAttrs.get("rows");
    if (!tblgen_rows)
      return emitError(loc, "'vector.flat_transpose' op ""requires attribute 'rows'");

    if (tblgen_rows && !(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
      return emitError(loc, "'vector.flat_transpose' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
    auto tblgen_columns = odsAttrs.get("columns");
    if (!tblgen_columns)
      return emitError(loc, "'vector.flat_transpose' op ""requires attribute 'columns'");

    if (tblgen_columns && !(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
      return emitError(loc, "'vector.flat_transpose' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FlatTransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FlatTransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FlatTransposeOp::getMatrix() {
  return *getODSOperands(0).begin();
}

::mlir::Value FlatTransposeOp::matrix() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FlatTransposeOp::getMatrixMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FlatTransposeOp::matrixMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FlatTransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FlatTransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FlatTransposeOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::Value FlatTransposeOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr FlatTransposeOp::getRowsAttr() {
  return (*this)->getAttr(getRowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t FlatTransposeOp::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr FlatTransposeOp::rowsAttr() {
  return (*this)->getAttr(rowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t FlatTransposeOp::rows() {
  auto attr = rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr FlatTransposeOp::getColumnsAttr() {
  return (*this)->getAttr(getColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t FlatTransposeOp::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr FlatTransposeOp::columnsAttr() {
  return (*this)->getAttr(columnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t FlatTransposeOp::columns() {
  auto attr = columnsAttr();
  return attr.getValue().getZExtValue();
}

void FlatTransposeOp::setRowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getRowsAttrName(), attr);
}

void FlatTransposeOp::rowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(rowsAttrName(), attr);
}

void FlatTransposeOp::setColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getColumnsAttrName(), attr);
}

void FlatTransposeOp::columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(columnsAttrName(), attr);
}

void FlatTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  odsState.addTypes(res);
}

void FlatTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FlatTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  odsState.addTypes(res);
}

void FlatTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FlatTransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FlatTransposeOp::verifyInvariantsImpl() {
  {
    auto tblgen_rows = (*this)->getAttr(getRowsAttrName());
    if (!tblgen_rows)
      return emitOpError("requires attribute 'rows'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps4(*this, tblgen_rows, "rows")))
      return ::mlir::failure();
  }
  {
    auto tblgen_columns = (*this)->getAttr(getColumnsAttrName());
    if (!tblgen_columns)
      return emitOpError("requires attribute 'columns'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps4(*this, tblgen_columns, "columns")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that source operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult FlatTransposeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FlatTransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType matrixRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> matrixOperands(matrixRawOperands);  ::llvm::SMLoc matrixOperandsLoc;
  (void)matrixOperandsLoc;
  ::mlir::Type matrixRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixTypes(matrixRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  matrixOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    matrixRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(matrixOperands, matrixTypes, matrixOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FlatTransposeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMatrix();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMatrix().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
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

void FlatTransposeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::FlatTransposeOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::GatherOp definitions
//===----------------------------------------------------------------------===//

GatherOpAdaptor::GatherOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

GatherOpAdaptor::GatherOpAdaptor(GatherOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange GatherOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GatherOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange GatherOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value GatherOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange GatherOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange GatherOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value GatherOpAdaptor::getIndexVec() {
  return *getODSOperands(2).begin();
}

::mlir::Value GatherOpAdaptor::index_vec() {
  return *getODSOperands(2).begin();
}

::mlir::Value GatherOpAdaptor::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value GatherOpAdaptor::mask() {
  return *getODSOperands(3).begin();
}

::mlir::Value GatherOpAdaptor::getPassThru() {
  return *getODSOperands(4).begin();
}

::mlir::Value GatherOpAdaptor::pass_thru() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr GatherOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GatherOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GatherOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range GatherOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value GatherOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range GatherOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range GatherOp::indices() {
  return getODSOperands(1);
}

::mlir::Value GatherOp::getIndexVec() {
  return *getODSOperands(2).begin();
}

::mlir::Value GatherOp::index_vec() {
  return *getODSOperands(2).begin();
}

::mlir::Value GatherOp::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value GatherOp::mask() {
  return *getODSOperands(3).begin();
}

::mlir::Value GatherOp::getPassThru() {
  return *getODSOperands(4).begin();
}

::mlir::Value GatherOp::pass_thru() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange GatherOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::getIndexVecMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::index_vecMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::getPassThruMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GatherOp::pass_thruMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GatherOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GatherOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GatherOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value GatherOp::result() {
  return *getODSResults(0).begin();
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value pass_thru) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(index_vec);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addTypes(result);
}

void GatherOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value pass_thru) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(index_vec);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GatherOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GatherOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GatherOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GatherOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::OpAsmParser::OperandType index_vecRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> index_vecOperands(index_vecRawOperands);  ::llvm::SMLoc index_vecOperandsLoc;
  (void)index_vecOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType pass_thruRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> pass_thruOperands(pass_thruRawOperands);  ::llvm::SMLoc pass_thruOperandsLoc;
  (void)pass_thruOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type index_vecRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> index_vecTypes(index_vecRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type pass_thruRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> pass_thruTypes(pass_thruRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  index_vecOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(index_vecRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  pass_thruOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(pass_thruRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    index_vecRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    pass_thruRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(index_vecOperands, index_vecTypes, index_vecOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(pass_thruOperands, pass_thruTypes, pass_thruOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GatherOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ' ' << "[";
  _odsPrinter << getIndexVec();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPassThru();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getIndexVec().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getPassThru().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void GatherOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::GatherOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertElementOp definitions
//===----------------------------------------------------------------------===//

InsertElementOpAdaptor::InsertElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

InsertElementOpAdaptor::InsertElementOpAdaptor(InsertElementOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange InsertElementOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange InsertElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOpAdaptor::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOpAdaptor::getPosition() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value InsertElementOpAdaptor::position() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr InsertElementOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InsertElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertElementOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InsertElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOp::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOp::getPosition() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value InsertElementOp::position() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange InsertElementOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::getPositionMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::positionMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value InsertElementOp::result() {
  return *getODSResults(0).begin();
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value position) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (position)
    odsState.addOperands(position);
  odsState.addTypes(result);
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value position) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (position)
    odsState.addOperands(position);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertElementOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value position) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  if (position)
    odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertElementOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSResults(0).begin()).getType().cast<ShapedType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that source operand type matches element type of result");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {dest, result} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertElementOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertElementOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = operands[1].getType();
  return ::mlir::success();
}

::mlir::ParseResult InsertElementOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> positionOperands;
  ::llvm::SMLoc positionOperandsLoc;
  (void)positionOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> positionTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  {
    positionOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      positionOperands.push_back(operand);
    }
  }
  if (!positionOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      positionTypes.push_back(optionalType);
    }
  }
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!(((type.isa<::mlir::VectorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be vector of any type values, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, resultTypes[0].cast<ShapedType>().getElementType(), sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, resultTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(positionOperands, positionTypes, positionOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertElementOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter << "[";
  if (getPosition()) {
  if (::mlir::Value value = getPosition())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (getPosition() ? ::llvm::ArrayRef<::mlir::Type>(getPosition().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  }
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertElementOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::InsertElementOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertMapOp definitions
//===----------------------------------------------------------------------===//

InsertMapOpAdaptor::InsertMapOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

InsertMapOpAdaptor::InsertMapOpAdaptor(InsertMapOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange InsertMapOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertMapOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange InsertMapOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertMapOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertMapOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertMapOpAdaptor::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertMapOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange InsertMapOpAdaptor::getIds() {
  return getODSOperands(2);
}

::mlir::ValueRange InsertMapOpAdaptor::ids() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr InsertMapOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InsertMapOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertMapOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InsertMapOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertMapOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertMapOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertMapOp::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertMapOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range InsertMapOp::getIds() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range InsertMapOp::ids() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InsertMapOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertMapOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertMapOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertMapOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertMapOp::getIdsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertMapOp::idsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertMapOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertMapOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertMapOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value InsertMapOp::result() {
  return *getODSResults(0).begin();
}

void InsertMapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::Value dest, ::mlir::ValueRange ids) {
  odsState.addOperands(vector);
  odsState.addOperands(dest);
  odsState.addOperands(ids);
  odsState.addTypes(result);
}

void InsertMapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value vector, ::mlir::Value dest, ::mlir::ValueRange ids) {
  odsState.addOperands(vector);
  odsState.addOperands(dest);
  odsState.addOperands(ids);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertMapOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertMapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value dest, ::mlir::ValueRange ids) {
  odsState.addOperands(vector);
  odsState.addOperands(dest);
  odsState.addOperands(ids);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertMapOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertMapOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertMapOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertMapOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {dest, result} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertMapOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertMapOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = operands[1].getType();
  return ::mlir::success();
}

::mlir::ParseResult InsertMapOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::OpAsmParser::OperandType destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> idsOperands;
  ::llvm::SMLoc idsOperandsLoc;
  (void)idsOperandsLoc;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  idsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(idsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, resultTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(idsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertMapOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter << "[";
  _odsPrinter << getIds();
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertMapOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::InsertMapOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertOp definitions
//===----------------------------------------------------------------------===//

InsertOpAdaptor::InsertOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

InsertOpAdaptor::InsertOpAdaptor(InsertOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange InsertOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InsertOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertOpAdaptor::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr InsertOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr InsertOpAdaptor::getPositionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("position").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertOpAdaptor::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

::mlir::ArrayAttr InsertOpAdaptor::positionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("position").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertOpAdaptor::position() {
  auto attr = positionAttr();
  return attr;
}

::mlir::LogicalResult InsertOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_position = odsAttrs.get("position");
    if (!tblgen_position)
      return emitError(loc, "'vector.insert' op ""requires attribute 'position'");

    if (tblgen_position && !(((tblgen_position.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_position.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.insert' op ""attribute 'position' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertOp::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange InsertOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
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

::mlir::Value InsertOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::Value InsertOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InsertOp::getPositionAttr() {
  return (*this)->getAttr(getPositionAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertOp::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

::mlir::ArrayAttr InsertOp::positionAttr() {
  return (*this)->getAttr(positionAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertOp::position() {
  auto attr = positionAttr();
  return attr;
}

void InsertOp::setPositionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPositionAttrName(), attr);
}

void InsertOp::positionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(positionAttrName(), attr);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr position) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  odsState.addTypes(res);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr position) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr position) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
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
    auto tblgen_position = (*this)->getAttr(getPositionAttrName());
    if (!tblgen_position)
      return emitOpError("requires attribute 'position'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_position, "position")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that source operand and result have same element type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {dest, res} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = operands[1].getType();
  return ::mlir::success();
}

::mlir::ParseResult InsertOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::ArrayAttr positionAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  destOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(positionAttr, parser.getBuilder().getType<::mlir::NoneType>(), "position",
          result.attributes)) {
    return ::mlir::failure();
  }
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
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addTypes(destTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getPositionAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"position"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::InsertOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::InsertStridedSliceOp definitions
//===----------------------------------------------------------------------===//

InsertStridedSliceOpAdaptor::InsertStridedSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

InsertStridedSliceOpAdaptor::InsertStridedSliceOpAdaptor(InsertStridedSliceOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange InsertStridedSliceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertStridedSliceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InsertStridedSliceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertStridedSliceOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertStridedSliceOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertStridedSliceOpAdaptor::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertStridedSliceOpAdaptor::dest() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr InsertStridedSliceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::getOffsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::offsetsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("offsets").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::offsets() {
  auto attr = offsetsAttr();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::getStridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::getStrides() {
  auto attr = getStridesAttr();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::stridesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("strides").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOpAdaptor::strides() {
  auto attr = stridesAttr();
  return attr;
}

::mlir::LogicalResult InsertStridedSliceOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_offsets = odsAttrs.get("offsets");
    if (!tblgen_offsets)
      return emitError(loc, "'vector.insert_strided_slice' op ""requires attribute 'offsets'");

    if (tblgen_offsets && !(((tblgen_offsets.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_offsets.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.insert_strided_slice' op ""attribute 'offsets' failed to satisfy constraint: 64-bit integer array attribute");
  }
  {
    auto tblgen_strides = odsAttrs.get("strides");
    if (!tblgen_strides)
      return emitError(loc, "'vector.insert_strided_slice' op ""requires attribute 'strides'");

    if (tblgen_strides && !(((tblgen_strides.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_strides.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.insert_strided_slice' op ""attribute 'strides' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertStridedSliceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertStridedSliceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertStridedSliceOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertStridedSliceOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertStridedSliceOp::getDest() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertStridedSliceOp::dest() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange InsertStridedSliceOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertStridedSliceOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertStridedSliceOp::getDestMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertStridedSliceOp::destMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertStridedSliceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertStridedSliceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertStridedSliceOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::Value InsertStridedSliceOp::res() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InsertStridedSliceOp::getOffsetsAttr() {
  return (*this)->getAttr(getOffsetsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertStridedSliceOp::getOffsets() {
  auto attr = getOffsetsAttr();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOp::offsetsAttr() {
  return (*this)->getAttr(offsetsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertStridedSliceOp::offsets() {
  auto attr = offsetsAttr();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOp::getStridesAttr() {
  return (*this)->getAttr(getStridesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertStridedSliceOp::getStrides() {
  auto attr = getStridesAttr();
  return attr;
}

::mlir::ArrayAttr InsertStridedSliceOp::stridesAttr() {
  return (*this)->getAttr(stridesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertStridedSliceOp::strides() {
  auto attr = stridesAttr();
  return attr;
}

void InsertStridedSliceOp::setOffsetsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOffsetsAttrName(), attr);
}

void InsertStridedSliceOp::offsetsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(offsetsAttrName(), attr);
}

void InsertStridedSliceOp::setStridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStridesAttrName(), attr);
}

void InsertStridedSliceOp::stridesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(stridesAttrName(), attr);
}

void InsertStridedSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
  odsState.addAttribute(getStridesAttrName(odsState.name), strides);
  odsState.addTypes(res);
}

void InsertStridedSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
  odsState.addAttribute(getStridesAttrName(odsState.name), strides);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(InsertStridedSliceOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void InsertStridedSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ArrayAttr offsets, ::mlir::ArrayAttr strides) {
  odsState.addOperands(source);
  odsState.addOperands(dest);
  odsState.addAttribute(getOffsetsAttrName(odsState.name), offsets);
  odsState.addAttribute(getStridesAttrName(odsState.name), strides);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertStridedSliceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InsertStridedSliceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(InsertStridedSliceOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult InsertStridedSliceOp::verifyInvariantsImpl() {
  {
    auto tblgen_offsets = (*this)->getAttr(getOffsetsAttrName());
    if (!tblgen_offsets)
      return emitOpError("requires attribute 'offsets'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_offsets, "offsets")))
      return ::mlir::failure();
  }
  {
    auto tblgen_strides = (*this)->getAttr(getStridesAttrName());
    if (!tblgen_strides)
      return emitOpError("requires attribute 'strides'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_strides, "strides")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that operand #0 and result have same element type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {dest, res} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult InsertStridedSliceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult InsertStridedSliceOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = operands[1].getType();
  return ::mlir::success();
}

::mlir::ParseResult InsertStridedSliceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType destRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> destOperands(destRawOperands);  ::llvm::SMLoc destOperandsLoc;
  (void)destOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
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
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destRawTypes[0] = type;
  }
  result.addTypes(destTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destOperands, destTypes, destOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InsertStridedSliceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getDest().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void InsertStridedSliceOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::InsertStridedSliceOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::LoadOp definitions
//===----------------------------------------------------------------------===//

LoadOpAdaptor::LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

LoadOpAdaptor::LoadOpAdaptor(LoadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange LoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value LoadOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange LoadOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange LoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr LoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
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

::mlir::Value LoadOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value LoadOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range LoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range LoadOp::indices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange LoadOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LoadOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LoadOp::indicesMutable() {
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
  return *getODSResults(0).begin();
}

::mlir::Value LoadOp::result() {
  return *getODSResults(0).begin();
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
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

::mlir::LogicalResult LoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
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
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::LoadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MaskedLoadOp definitions
//===----------------------------------------------------------------------===//

MaskedLoadOpAdaptor::MaskedLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MaskedLoadOpAdaptor::MaskedLoadOpAdaptor(MaskedLoadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MaskedLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskedLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange MaskedLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedLoadOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange MaskedLoadOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange MaskedLoadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value MaskedLoadOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedLoadOpAdaptor::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedLoadOpAdaptor::getPassThru() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskedLoadOpAdaptor::pass_thru() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MaskedLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskedLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaskedLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
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

::mlir::Operation::operand_range MaskedLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedLoadOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range MaskedLoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MaskedLoadOp::indices() {
  return getODSOperands(1);
}

::mlir::Value MaskedLoadOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedLoadOp::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedLoadOp::getPassThru() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskedLoadOp::pass_thru() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MaskedLoadOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::getPassThruMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::pass_thruMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaskedLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskedLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value MaskedLoadOp::result() {
  return *getODSResults(0).begin();
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addTypes(result);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value pass_thru) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskedLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaskedLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult MaskedLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType pass_thruRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> pass_thruOperands(pass_thruRawOperands);  ::llvm::SMLoc pass_thruOperandsLoc;
  (void)pass_thruOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type pass_thruRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> pass_thruTypes(pass_thruRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  pass_thruOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(pass_thruRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    pass_thruRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(pass_thruOperands, pass_thruTypes, pass_thruOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskedLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPassThru();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getPassThru().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MaskedLoadOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::MaskedLoadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MaskedStoreOp definitions
//===----------------------------------------------------------------------===//

MaskedStoreOpAdaptor::MaskedStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MaskedStoreOpAdaptor::MaskedStoreOpAdaptor(MaskedStoreOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MaskedStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskedStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange MaskedStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedStoreOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedStoreOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange MaskedStoreOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange MaskedStoreOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value MaskedStoreOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedStoreOpAdaptor::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedStoreOpAdaptor::getValueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskedStoreOpAdaptor::valueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MaskedStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaskedStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaskedStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false};
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

::mlir::Operation::operand_range MaskedStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedStoreOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedStoreOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range MaskedStoreOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MaskedStoreOp::indices() {
  return getODSOperands(1);
}

::mlir::Value MaskedStoreOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedStoreOp::mask() {
  return *getODSOperands(2).begin();
}

::mlir::Value MaskedStoreOp::getValueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::Value MaskedStoreOp::valueToStore() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MaskedStoreOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::getValueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::valueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaskedStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskedStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(valueToStore);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value mask, ::mlir::Value valueToStore) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(mask);
  odsState.addOperands(valueToStore);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskedStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaskedStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult MaskedStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType valueToStoreRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueToStoreOperands(valueToStoreRawOperands);  ::llvm::SMLoc valueToStoreOperandsLoc;
  (void)valueToStoreOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type valueToStoreRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueToStoreTypes(valueToStoreRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valueToStoreOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueToStoreRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueToStoreRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueToStoreOperands, valueToStoreTypes, valueToStoreOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskedStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getValueToStore();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getValueToStore().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MaskedStoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::MaskedStoreOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MatmulOp definitions
//===----------------------------------------------------------------------===//

MatmulOpAdaptor::MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MatmulOpAdaptor::MatmulOpAdaptor(MatmulOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MatmulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatmulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatmulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatmulOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatmulOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatmulOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value MatmulOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MatmulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatmulOpAdaptor::getLhsRowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("lhs_rows").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MatmulOpAdaptor::getLhsRows() {
  auto attr = getLhsRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOpAdaptor::lhs_rowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("lhs_rows").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MatmulOpAdaptor::lhs_rows() {
  auto attr = lhs_rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOpAdaptor::getLhsColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("lhs_columns").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MatmulOpAdaptor::getLhsColumns() {
  auto attr = getLhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOpAdaptor::lhs_columnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("lhs_columns").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MatmulOpAdaptor::lhs_columns() {
  auto attr = lhs_columnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOpAdaptor::getRhsColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rhs_columns").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MatmulOpAdaptor::getRhsColumns() {
  auto attr = getRhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOpAdaptor::rhs_columnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("rhs_columns").cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MatmulOpAdaptor::rhs_columns() {
  auto attr = rhs_columnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MatmulOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_lhs_rows = odsAttrs.get("lhs_rows");
    if (!tblgen_lhs_rows)
      return emitError(loc, "'vector.matrix_multiply' op ""requires attribute 'lhs_rows'");

    if (tblgen_lhs_rows && !(((tblgen_lhs_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_lhs_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
      return emitError(loc, "'vector.matrix_multiply' op ""attribute 'lhs_rows' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
    auto tblgen_lhs_columns = odsAttrs.get("lhs_columns");
    if (!tblgen_lhs_columns)
      return emitError(loc, "'vector.matrix_multiply' op ""requires attribute 'lhs_columns'");

    if (tblgen_lhs_columns && !(((tblgen_lhs_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_lhs_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
      return emitError(loc, "'vector.matrix_multiply' op ""attribute 'lhs_columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  {
    auto tblgen_rhs_columns = odsAttrs.get("rhs_columns");
    if (!tblgen_rhs_columns)
      return emitError(loc, "'vector.matrix_multiply' op ""requires attribute 'rhs_columns'");

    if (tblgen_rhs_columns && !(((tblgen_rhs_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_rhs_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
      return emitError(loc, "'vector.matrix_multiply' op ""attribute 'rhs_columns' failed to satisfy constraint: 32-bit signless integer attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatmulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatmulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatmulOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatmulOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatmulOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value MatmulOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatmulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatmulOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatmulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatmulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatmulOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::Value MatmulOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatmulOp::getLhsRowsAttr() {
  return (*this)->getAttr(getLhsRowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatmulOp::getLhsRows() {
  auto attr = getLhsRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOp::lhs_rowsAttr() {
  return (*this)->getAttr(lhs_rowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatmulOp::lhs_rows() {
  auto attr = lhs_rowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOp::getLhsColumnsAttr() {
  return (*this)->getAttr(getLhsColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatmulOp::getLhsColumns() {
  auto attr = getLhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOp::lhs_columnsAttr() {
  return (*this)->getAttr(lhs_columnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatmulOp::lhs_columns() {
  auto attr = lhs_columnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOp::getRhsColumnsAttr() {
  return (*this)->getAttr(getRhsColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatmulOp::getRhsColumns() {
  auto attr = getRhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatmulOp::rhs_columnsAttr() {
  return (*this)->getAttr(rhs_columnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatmulOp::rhs_columns() {
  auto attr = rhs_columnsAttr();
  return attr.getValue().getZExtValue();
}

void MatmulOp::setLhsRowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLhsRowsAttrName(), attr);
}

void MatmulOp::lhs_rowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(lhs_rowsAttrName(), attr);
}

void MatmulOp::setLhsColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLhsColumnsAttrName(), attr);
}

void MatmulOp::lhs_columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(lhs_columnsAttrName(), attr);
}

void MatmulOp::setRhsColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getRhsColumnsAttrName(), attr);
}

void MatmulOp::rhs_columnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(rhs_columnsAttrName(), attr);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs, unsigned lhsRows, unsigned lhsColumns, unsigned rhsColumns) {
     odsState.addOperands({lhs, rhs});
     odsState.addAttribute("lhs_rows",odsBuilder.getI32IntegerAttr(lhsRows));
     odsState.addAttribute("lhs_columns",odsBuilder.getI32IntegerAttr(lhsColumns));
     odsState.addAttribute("rhs_columns",odsBuilder.getI32IntegerAttr(rhsColumns));
     odsState.addTypes(VectorType::get(lhsRows * rhsColumns,
       lhs.getType().cast<VectorType>().getElementType()));
   
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), lhs_rows);
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), lhs_columns);
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), rhs_columns);
  odsState.addTypes(res);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), lhs_rows);
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), lhs_columns);
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), rhs_columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_rows));
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_columns));
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rhs_columns));
  odsState.addTypes(res);
}

void MatmulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_rows));
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_columns));
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rhs_columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatmulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatmulOp::verifyInvariantsImpl() {
  {
    auto tblgen_lhs_rows = (*this)->getAttr(getLhsRowsAttrName());
    if (!tblgen_lhs_rows)
      return emitOpError("requires attribute 'lhs_rows'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps4(*this, tblgen_lhs_rows, "lhs_rows")))
      return ::mlir::failure();
  }
  {
    auto tblgen_lhs_columns = (*this)->getAttr(getLhsColumnsAttrName());
    if (!tblgen_lhs_columns)
      return emitOpError("requires attribute 'lhs_columns'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps4(*this, tblgen_lhs_columns, "lhs_columns")))
      return ::mlir::failure();
  }
  {
    auto tblgen_rhs_columns = (*this)->getAttr(getRhsColumnsAttrName());
    if (!tblgen_rhs_columns)
      return emitOpError("requires attribute 'rhs_columns'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps4(*this, tblgen_rhs_columns, "rhs_columns")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that lhs operand and result have same element type");
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(1)))))
    return emitOpError("failed to verify that rhs operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult MatmulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatmulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::OperandType rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
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
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
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
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatmulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
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

void MatmulOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::MatmulOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::MultiDimReductionOp definitions
//===----------------------------------------------------------------------===//

MultiDimReductionOpAdaptor::MultiDimReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MultiDimReductionOpAdaptor::MultiDimReductionOpAdaptor(MultiDimReductionOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange MultiDimReductionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MultiDimReductionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MultiDimReductionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MultiDimReductionOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value MultiDimReductionOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MultiDimReductionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::vector::CombiningKindAttr MultiDimReductionOpAdaptor::getKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").cast<::mlir::vector::CombiningKindAttr>();
  return attr;
}

::mlir::vector::CombiningKind MultiDimReductionOpAdaptor::getKind() {
  auto attr = getKindAttr();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr MultiDimReductionOpAdaptor::kindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").cast<::mlir::vector::CombiningKindAttr>();
  return attr;
}

::mlir::vector::CombiningKind MultiDimReductionOpAdaptor::kind() {
  auto attr = kindAttr();
  return attr.getKind();
}

::mlir::ArrayAttr MultiDimReductionOpAdaptor::getReductionDimsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("reduction_dims").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MultiDimReductionOpAdaptor::getReductionDims() {
  auto attr = getReductionDimsAttr();
  return attr;
}

::mlir::ArrayAttr MultiDimReductionOpAdaptor::reduction_dimsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("reduction_dims").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr MultiDimReductionOpAdaptor::reduction_dims() {
  auto attr = reduction_dimsAttr();
  return attr;
}

::mlir::LogicalResult MultiDimReductionOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_kind = odsAttrs.get("kind");
    if (!tblgen_kind)
      return emitError(loc, "'vector.multi_reduction' op ""requires attribute 'kind'");

    if (tblgen_kind && !((tblgen_kind.isa<::mlir::vector::CombiningKindAttr>())))
      return emitError(loc, "'vector.multi_reduction' op ""attribute 'kind' failed to satisfy constraint: Kind of combining function for contractions and reductions");
  }
  {
    auto tblgen_reduction_dims = odsAttrs.get("reduction_dims");
    if (!tblgen_reduction_dims)
      return emitError(loc, "'vector.multi_reduction' op ""requires attribute 'reduction_dims'");

    if (tblgen_reduction_dims && !(((tblgen_reduction_dims.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reduction_dims.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.multi_reduction' op ""attribute 'reduction_dims' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MultiDimReductionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MultiDimReductionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MultiDimReductionOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value MultiDimReductionOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MultiDimReductionOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MultiDimReductionOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MultiDimReductionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MultiDimReductionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MultiDimReductionOp::getDest() {
  return *getODSResults(0).begin();
}

::mlir::Value MultiDimReductionOp::dest() {
  return *getODSResults(0).begin();
}

::mlir::vector::CombiningKindAttr MultiDimReductionOp::getKindAttr() {
  return (*this)->getAttr(getKindAttrName()).cast<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind MultiDimReductionOp::getKind() {
  auto attr = getKindAttr();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr MultiDimReductionOp::kindAttr() {
  return (*this)->getAttr(kindAttrName()).cast<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind MultiDimReductionOp::kind() {
  auto attr = kindAttr();
  return attr.getKind();
}

::mlir::ArrayAttr MultiDimReductionOp::getReductionDimsAttr() {
  return (*this)->getAttr(getReductionDimsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MultiDimReductionOp::getReductionDims() {
  auto attr = getReductionDimsAttr();
  return attr;
}

::mlir::ArrayAttr MultiDimReductionOp::reduction_dimsAttr() {
  return (*this)->getAttr(reduction_dimsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr MultiDimReductionOp::reduction_dims() {
  auto attr = reduction_dimsAttr();
  return attr;
}

void MultiDimReductionOp::setKindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(getKindAttrName(), attr);
}

void MultiDimReductionOp::kindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(kindAttrName(), attr);
}

void MultiDimReductionOp::setReductionDimsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getReductionDimsAttrName(), attr);
}

void MultiDimReductionOp::reduction_dimsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(reduction_dimsAttrName(), attr);
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims) {
  odsState.addOperands(source);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addAttribute(getReductionDimsAttrName(odsState.name), reduction_dims);
  odsState.addTypes(dest);
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims) {
  odsState.addOperands(source);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addAttribute(getReductionDimsAttrName(odsState.name), reduction_dims);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MultiDimReductionOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims) {
  odsState.addOperands(source);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addAttribute(getReductionDimsAttrName(odsState.name), reduction_dims);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims) {
  odsState.addOperands(source);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addAttribute(getReductionDimsAttrName(odsState.name), reduction_dims);
  odsState.addTypes(dest);
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims) {
  odsState.addOperands(source);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addAttribute(getReductionDimsAttrName(odsState.name), reduction_dims);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MultiDimReductionOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::ArrayAttr reduction_dims) {
  odsState.addOperands(source);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addAttribute(getReductionDimsAttrName(odsState.name), reduction_dims);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MultiDimReductionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MultiDimReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MultiDimReductionOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MultiDimReductionOp::verifyInvariantsImpl() {
  {
    auto tblgen_kind = (*this)->getAttr(getKindAttrName());
    if (!tblgen_kind)
      return emitOpError("requires attribute 'kind'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps3(*this, tblgen_kind, "kind")))
      return ::mlir::failure();
  }
  {
    auto tblgen_reduction_dims = (*this)->getAttr(getReductionDimsAttrName());
    if (!tblgen_reduction_dims)
      return emitOpError("requires attribute 'reduction_dims'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_reduction_dims, "reduction_dims")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that source operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult MultiDimReductionOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MultiDimReductionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::vector::CombiningKindAttr kindAttr;
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::ArrayAttr reduction_dimsAttr;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type destRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destTypes(destRawTypes);

  if (parser.parseCustomAttributeWithFallback(kindAttr, ::mlir::Type{}, "kind",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(reduction_dimsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "reduction_dims",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
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

void MultiDimReductionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getKindAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"kind", "reduction_dims"});
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getReductionDimsAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
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

void MultiDimReductionOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::MultiDimReductionOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::OuterProductOp definitions
//===----------------------------------------------------------------------===//

OuterProductOpAdaptor::OuterProductOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

OuterProductOpAdaptor::OuterProductOpAdaptor(OuterProductOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange OuterProductOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OuterProductOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange OuterProductOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OuterProductOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OuterProductOpAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OuterProductOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value OuterProductOpAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange OuterProductOpAdaptor::getAcc() {
  return getODSOperands(2);
}

::mlir::ValueRange OuterProductOpAdaptor::acc() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr OuterProductOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::vector::CombiningKindAttr OuterProductOpAdaptor::getKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
  if (!attr)
    attr = ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext());
  return attr;
}

::mlir::vector::CombiningKind OuterProductOpAdaptor::getKind() {
  auto attr = getKindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext()).getKind();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr OuterProductOpAdaptor::kindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
  if (!attr)
    attr = ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext());
  return attr;
}

::mlir::vector::CombiningKind OuterProductOpAdaptor::kind() {
  auto attr = kindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder(odsAttrs.getContext()).getContext()).getKind();
  return attr.getKind();
}

::mlir::LogicalResult OuterProductOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_kind = odsAttrs.get("kind");
    if (tblgen_kind && !((tblgen_kind.isa<::mlir::vector::CombiningKindAttr>())))
      return emitError(loc, "'vector.outerproduct' op ""attribute 'kind' failed to satisfy constraint: Kind of combining function for contractions and reductions");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OuterProductOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OuterProductOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OuterProductOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OuterProductOp::lhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OuterProductOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value OuterProductOp::rhs() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range OuterProductOp::getAcc() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range OuterProductOp::acc() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange OuterProductOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OuterProductOp::lhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OuterProductOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OuterProductOp::rhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OuterProductOp::getAccMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OuterProductOp::accMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OuterProductOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OuterProductOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::vector::CombiningKindAttr OuterProductOp::getKindAttr() {
  return (*this)->getAttr(getKindAttrName()).dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind OuterProductOp::getKind() {
  auto attr = getKindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder((*this)->getContext()).getContext()).getKind();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr OuterProductOp::kindAttr() {
  return (*this)->getAttr(kindAttrName()).dyn_cast_or_null<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind OuterProductOp::kind() {
  auto attr = kindAttr();
    if (!attr)
      return ::mlir::vector::CombiningKindAttr::get(CombiningKind::ADD, ::mlir::Builder((*this)->getContext()).getContext()).getKind();
  return attr.getKind();
}

void OuterProductOp::setKindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(getKindAttrName(), attr);
}

void OuterProductOp::kindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(kindAttrName(), attr);
}

void OuterProductOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKindAttr kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addTypes(resultType0);
}

void OuterProductOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKindAttr kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OuterProductOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKind kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addTypes(resultType0);
}

void OuterProductOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::ValueRange acc, ::mlir::vector::CombiningKind kind) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OuterProductOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OuterProductOp::verifyInvariantsImpl() {
  {
    auto tblgen_kind = (*this)->getAttr(getKindAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps3(*this, tblgen_kind, "kind")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that lhs operand and result have same element type");
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(1)))))
    return emitOpError("failed to verify that rhs operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult OuterProductOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void OuterProductOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::OuterProductOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::PrintOp definitions
//===----------------------------------------------------------------------===//

PrintOpAdaptor::PrintOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

PrintOpAdaptor::PrintOpAdaptor(PrintOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange PrintOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PrintOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PrintOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrintOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value PrintOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PrintOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PrintOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PrintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrintOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value PrintOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PrintOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PrintOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PrintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PrintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source) {
  odsState.addOperands(source);
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PrintOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PrintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);

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
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::PrintOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ReductionOp definitions
//===----------------------------------------------------------------------===//

ReductionOpAdaptor::ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ReductionOpAdaptor::ReductionOpAdaptor(ReductionOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ReductionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReductionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReductionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOpAdaptor::getAcc() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ReductionOpAdaptor::acc() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr ReductionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::vector::CombiningKindAttr ReductionOpAdaptor::getKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").cast<::mlir::vector::CombiningKindAttr>();
  return attr;
}

::mlir::vector::CombiningKind ReductionOpAdaptor::getKind() {
  auto attr = getKindAttr();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr ReductionOpAdaptor::kindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").cast<::mlir::vector::CombiningKindAttr>();
  return attr;
}

::mlir::vector::CombiningKind ReductionOpAdaptor::kind() {
  auto attr = kindAttr();
  return attr.getKind();
}

::mlir::LogicalResult ReductionOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_kind = odsAttrs.get("kind");
    if (!tblgen_kind)
      return emitError(loc, "'vector.reduction' op ""requires attribute 'kind'");

    if (tblgen_kind && !((tblgen_kind.isa<::mlir::vector::CombiningKindAttr>())))
      return emitError(loc, "'vector.reduction' op ""attribute 'kind' failed to satisfy constraint: Kind of combining function for contractions and reductions");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReductionOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReductionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOp::getAcc() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ReductionOp::acc() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange ReductionOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReductionOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReductionOp::getAccMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReductionOp::accMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReductionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOp::getDest() {
  return *getODSResults(0).begin();
}

::mlir::Value ReductionOp::dest() {
  return *getODSResults(0).begin();
}

::mlir::vector::CombiningKindAttr ReductionOp::getKindAttr() {
  return (*this)->getAttr(getKindAttrName()).cast<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind ReductionOp::getKind() {
  auto attr = getKindAttr();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr ReductionOp::kindAttr() {
  return (*this)->getAttr(kindAttrName()).cast<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind ReductionOp::kind() {
  auto attr = kindAttr();
  return attr.getKind();
}

void ReductionOp::setKindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(getKindAttrName(), attr);
}

void ReductionOp::kindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(kindAttrName(), attr);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value vector, /*optional*/::mlir::Value acc) {
  odsState.addOperands(vector);
  if (acc)
    odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addTypes(dest);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value vector, /*optional*/::mlir::Value acc) {
  odsState.addOperands(vector);
  if (acc)
    odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::vector::CombiningKind kind, ::mlir::Value vector, /*optional*/::mlir::Value acc) {
  odsState.addOperands(vector);
  if (acc)
    odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addTypes(dest);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKind kind, ::mlir::Value vector, /*optional*/::mlir::Value acc) {
  odsState.addOperands(vector);
  if (acc)
    odsState.addOperands(acc);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionOp::verifyInvariantsImpl() {
  {
    auto tblgen_kind = (*this)->getAttr(getKindAttrName());
    if (!tblgen_kind)
      return emitOpError("requires attribute 'kind'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps3(*this, tblgen_kind, "kind")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that source operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult ReductionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ReductionOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ReductionOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ReshapeOp definitions
//===----------------------------------------------------------------------===//

ReshapeOpAdaptor::ReshapeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ReshapeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReshapeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReshapeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReshapeOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ReshapeOpAdaptor::getInputShape() {
  return getODSOperands(1);
}

::mlir::ValueRange ReshapeOpAdaptor::input_shape() {
  return getODSOperands(1);
}

::mlir::ValueRange ReshapeOpAdaptor::getOutputShape() {
  return getODSOperands(2);
}

::mlir::ValueRange ReshapeOpAdaptor::output_shape() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr ReshapeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ReshapeOpAdaptor::getFixedVectorSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("fixed_vector_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ReshapeOpAdaptor::getFixedVectorSizes() {
  auto attr = getFixedVectorSizesAttr();
  return attr;
}

::mlir::ArrayAttr ReshapeOpAdaptor::fixed_vector_sizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("fixed_vector_sizes").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ReshapeOpAdaptor::fixed_vector_sizes() {
  auto attr = fixed_vector_sizesAttr();
  return attr;
}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'vector.reshape' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'vector.reshape' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_fixed_vector_sizes = odsAttrs.get("fixed_vector_sizes");
    if (!tblgen_fixed_vector_sizes)
      return emitError(loc, "'vector.reshape' op ""requires attribute 'fixed_vector_sizes'");

    if (tblgen_fixed_vector_sizes && !(((tblgen_fixed_vector_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_fixed_vector_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.reshape' op ""attribute 'fixed_vector_sizes' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReshapeOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReshapeOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ReshapeOp::getInputShape() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ReshapeOp::input_shape() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ReshapeOp::getOutputShape() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ReshapeOp::output_shape() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ReshapeOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::getInputShapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::input_shapeMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::getOutputShapeMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ReshapeOp::output_shapeMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
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

::mlir::Value ReshapeOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value ReshapeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ReshapeOp::getFixedVectorSizesAttr() {
  return (*this)->getAttr(getFixedVectorSizesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ReshapeOp::getFixedVectorSizes() {
  auto attr = getFixedVectorSizesAttr();
  return attr;
}

::mlir::ArrayAttr ReshapeOp::fixed_vector_sizesAttr() {
  return (*this)->getAttr(fixed_vector_sizesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ReshapeOp::fixed_vector_sizes() {
  auto attr = fixed_vector_sizesAttr();
  return attr;
}

void ReshapeOp::setFixedVectorSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getFixedVectorSizesAttrName(), attr);
}

void ReshapeOp::fixed_vector_sizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(fixed_vector_sizesAttrName(), attr);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::ValueRange input_shape, ::mlir::ValueRange output_shape, ::mlir::ArrayAttr fixed_vector_sizes) {
  odsState.addOperands(vector);
  odsState.addOperands(input_shape);
  odsState.addOperands(output_shape);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(input_shape.size()), static_cast<int32_t>(output_shape.size())}));
  odsState.addAttribute(getFixedVectorSizesAttrName(odsState.name), fixed_vector_sizes);
  odsState.addTypes(result);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ValueRange input_shape, ::mlir::ValueRange output_shape, ::mlir::ArrayAttr fixed_vector_sizes) {
  odsState.addOperands(vector);
  odsState.addOperands(input_shape);
  odsState.addOperands(output_shape);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(input_shape.size()), static_cast<int32_t>(output_shape.size())}));
  odsState.addAttribute(getFixedVectorSizesAttrName(odsState.name), fixed_vector_sizes);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
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
    auto tblgen_fixed_vector_sizes = (*this)->getAttr(getFixedVectorSizesAttrName());
    if (!tblgen_fixed_vector_sizes)
      return emitOpError("requires attribute 'fixed_vector_sizes'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_fixed_vector_sizes, "fixed_vector_sizes")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
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
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> input_shapeOperands;
  ::llvm::SMLoc input_shapeOperandsLoc;
  (void)input_shapeOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> output_shapeOperands;
  ::llvm::SMLoc output_shapeOperandsLoc;
  (void)output_shapeOperandsLoc;
  ::mlir::ArrayAttr fixed_vector_sizesAttr;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  input_shapeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(input_shapeOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  output_shapeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(output_shapeOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(fixed_vector_sizesAttr, parser.getBuilder().getType<::mlir::NoneType>(), "fixed_vector_sizes",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(input_shapeOperands.size()), static_cast<int32_t>(output_shapeOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(input_shapeOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(output_shapeOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << ",";
  _odsPrinter << ' ' << "[";
  _odsPrinter << getInputShape();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ' << "[";
  _odsPrinter << getOutputShape();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getFixedVectorSizesAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "fixed_vector_sizes"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ReshapeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ReshapeOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ScanOp definitions
//===----------------------------------------------------------------------===//

ScanOpAdaptor::ScanOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ScanOpAdaptor::ScanOpAdaptor(ScanOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ScanOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScanOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScanOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScanOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScanOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScanOpAdaptor::getInitialValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScanOpAdaptor::initial_value() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ScanOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::vector::CombiningKindAttr ScanOpAdaptor::getKindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").cast<::mlir::vector::CombiningKindAttr>();
  return attr;
}

::mlir::vector::CombiningKind ScanOpAdaptor::getKind() {
  auto attr = getKindAttr();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr ScanOpAdaptor::kindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::vector::CombiningKindAttr attr = odsAttrs.get("kind").cast<::mlir::vector::CombiningKindAttr>();
  return attr;
}

::mlir::vector::CombiningKind ScanOpAdaptor::kind() {
  auto attr = kindAttr();
  return attr.getKind();
}

::mlir::IntegerAttr ScanOpAdaptor::getReductionDimAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("reduction_dim").cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ScanOpAdaptor::getReductionDim() {
  auto attr = getReductionDimAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ScanOpAdaptor::reduction_dimAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("reduction_dim").cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t ScanOpAdaptor::reduction_dim() {
  auto attr = reduction_dimAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ScanOpAdaptor::getInclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("inclusive").cast<::mlir::BoolAttr>();
  return attr;
}

bool ScanOpAdaptor::getInclusive() {
  auto attr = getInclusiveAttr();
  return attr.getValue();
}

::mlir::BoolAttr ScanOpAdaptor::inclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::BoolAttr attr = odsAttrs.get("inclusive").cast<::mlir::BoolAttr>();
  return attr;
}

bool ScanOpAdaptor::inclusive() {
  auto attr = inclusiveAttr();
  return attr.getValue();
}

::mlir::LogicalResult ScanOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_kind = odsAttrs.get("kind");
    if (!tblgen_kind)
      return emitError(loc, "'vector.scan' op ""requires attribute 'kind'");

    if (tblgen_kind && !((tblgen_kind.isa<::mlir::vector::CombiningKindAttr>())))
      return emitError(loc, "'vector.scan' op ""attribute 'kind' failed to satisfy constraint: Kind of combining function for contractions and reductions");
  }
  {
    auto tblgen_reduction_dim = odsAttrs.get("reduction_dim");
    if (!tblgen_reduction_dim)
      return emitError(loc, "'vector.scan' op ""requires attribute 'reduction_dim'");

    if (tblgen_reduction_dim && !(((tblgen_reduction_dim.isa<::mlir::IntegerAttr>())) && ((tblgen_reduction_dim.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
      return emitError(loc, "'vector.scan' op ""attribute 'reduction_dim' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
    auto tblgen_inclusive = odsAttrs.get("inclusive");
    if (!tblgen_inclusive)
      return emitError(loc, "'vector.scan' op ""requires attribute 'inclusive'");

    if (tblgen_inclusive && !((tblgen_inclusive.isa<::mlir::BoolAttr>())))
      return emitError(loc, "'vector.scan' op ""attribute 'inclusive' failed to satisfy constraint: bool attribute");
  }
  return ::mlir::success();
}

void ScanOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "dest");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "accumulated_value");
}

std::pair<unsigned, unsigned> ScanOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScanOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScanOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScanOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScanOp::getInitialValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value ScanOp::initial_value() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ScanOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScanOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScanOp::getInitialValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScanOp::initial_valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScanOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScanOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScanOp::getDest() {
  return *getODSResults(0).begin();
}

::mlir::Value ScanOp::dest() {
  return *getODSResults(0).begin();
}

::mlir::Value ScanOp::getAccumulatedValue() {
  return *getODSResults(1).begin();
}

::mlir::Value ScanOp::accumulated_value() {
  return *getODSResults(1).begin();
}

::mlir::vector::CombiningKindAttr ScanOp::getKindAttr() {
  return (*this)->getAttr(getKindAttrName()).cast<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind ScanOp::getKind() {
  auto attr = getKindAttr();
  return attr.getKind();
}

::mlir::vector::CombiningKindAttr ScanOp::kindAttr() {
  return (*this)->getAttr(kindAttrName()).cast<::mlir::vector::CombiningKindAttr>();
}

::mlir::vector::CombiningKind ScanOp::kind() {
  auto attr = kindAttr();
  return attr.getKind();
}

::mlir::IntegerAttr ScanOp::getReductionDimAttr() {
  return (*this)->getAttr(getReductionDimAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ScanOp::getReductionDim() {
  auto attr = getReductionDimAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ScanOp::reduction_dimAttr() {
  return (*this)->getAttr(reduction_dimAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ScanOp::reduction_dim() {
  auto attr = reduction_dimAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ScanOp::getInclusiveAttr() {
  return (*this)->getAttr(getInclusiveAttrName()).cast<::mlir::BoolAttr>();
}

bool ScanOp::getInclusive() {
  auto attr = getInclusiveAttr();
  return attr.getValue();
}

::mlir::BoolAttr ScanOp::inclusiveAttr() {
  return (*this)->getAttr(inclusiveAttrName()).cast<::mlir::BoolAttr>();
}

bool ScanOp::inclusive() {
  auto attr = inclusiveAttr();
  return attr.getValue();
}

void ScanOp::setKindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(getKindAttrName(), attr);
}

void ScanOp::kindAttr(::mlir::vector::CombiningKindAttr attr) {
  (*this)->setAttr(kindAttrName(), attr);
}

void ScanOp::setReductionDimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getReductionDimAttrName(), attr);
}

void ScanOp::reduction_dimAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(reduction_dimAttrName(), attr);
}

void ScanOp::setInclusiveAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getInclusiveAttrName(), attr);
}

void ScanOp::inclusiveAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(inclusiveAttrName(), attr);
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Type accumulated_value, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::Value initial_value, ::mlir::IntegerAttr reduction_dim, ::mlir::BoolAttr inclusive) {
  odsState.addOperands(source);
  odsState.addOperands(initial_value);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addAttribute(getReductionDimAttrName(odsState.name), reduction_dim);
  odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  odsState.addTypes(dest);
  odsState.addTypes(accumulated_value);
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::Value initial_value, ::mlir::IntegerAttr reduction_dim, ::mlir::BoolAttr inclusive) {
  odsState.addOperands(source);
  odsState.addOperands(initial_value);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addAttribute(getReductionDimAttrName(odsState.name), reduction_dim);
  odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ScanOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKindAttr kind, ::mlir::Value source, ::mlir::Value initial_value, ::mlir::IntegerAttr reduction_dim, ::mlir::BoolAttr inclusive) {
  odsState.addOperands(source);
  odsState.addOperands(initial_value);
  odsState.addAttribute(getKindAttrName(odsState.name), kind);
  odsState.addAttribute(getReductionDimAttrName(odsState.name), reduction_dim);
  odsState.addAttribute(getInclusiveAttrName(odsState.name), inclusive);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Type accumulated_value, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::Value initial_value, uint64_t reduction_dim, bool inclusive) {
  odsState.addOperands(source);
  odsState.addOperands(initial_value);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addAttribute(getReductionDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), reduction_dim));
  odsState.addAttribute(getInclusiveAttrName(odsState.name), odsBuilder.getBoolAttr(inclusive));
  odsState.addTypes(dest);
  odsState.addTypes(accumulated_value);
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::Value initial_value, uint64_t reduction_dim, bool inclusive) {
  odsState.addOperands(source);
  odsState.addOperands(initial_value);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addAttribute(getReductionDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), reduction_dim));
  odsState.addAttribute(getInclusiveAttrName(odsState.name), odsBuilder.getBoolAttr(inclusive));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ScanOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::vector::CombiningKind kind, ::mlir::Value source, ::mlir::Value initial_value, uint64_t reduction_dim, bool inclusive) {
  odsState.addOperands(source);
  odsState.addOperands(initial_value);
  odsState.addAttribute(getKindAttrName(odsState.name), ::mlir::vector::CombiningKindAttr::get(kind, odsBuilder.getContext()));
  odsState.addAttribute(getReductionDimAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), reduction_dim));
  odsState.addAttribute(getInclusiveAttrName(odsState.name), odsBuilder.getBoolAttr(inclusive));
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScanOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ScanOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ScanOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 2u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ScanOp::verifyInvariantsImpl() {
  {
    auto tblgen_kind = (*this)->getAttr(getKindAttrName());
    if (!tblgen_kind)
      return emitOpError("requires attribute 'kind'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps3(*this, tblgen_kind, "kind")))
      return ::mlir::failure();
  }
  {
    auto tblgen_reduction_dim = (*this)->getAttr(getReductionDimAttrName());
    if (!tblgen_reduction_dim)
      return emitOpError("requires attribute 'reduction_dim'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps5(*this, tblgen_reduction_dim, "reduction_dim")))
      return ::mlir::failure();
  }
  {
    auto tblgen_inclusive = (*this)->getAttr(getInclusiveAttrName());
    if (!tblgen_inclusive)
      return emitOpError("requires attribute 'inclusive'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps6(*this, tblgen_inclusive, "inclusive")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {source, dest} have same type");
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSResults(1).begin()).getType()})))))
    return emitOpError("failed to verify that all of {initial_value, accumulated_value} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult ScanOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult ScanOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(2);
  ::mlir::Builder odsBuilder(context);
  inferredReturnTypes[0] = operands[0].getType();
  inferredReturnTypes[1] = operands[1].getType();
  return ::mlir::success();
}

::mlir::ParseResult ScanOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::vector::CombiningKindAttr kindAttr;
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
  (void)sourceOperandsLoc;
  ::mlir::OpAsmParser::OperandType initial_valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> initial_valueOperands(initial_valueRawOperands);  ::llvm::SMLoc initial_valueOperandsLoc;
  (void)initial_valueOperandsLoc;
  ::mlir::Type sourceRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(sourceRawTypes);
  ::mlir::Type initial_valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> initial_valueTypes(initial_valueRawTypes);

  if (parser.parseCustomAttributeWithFallback(kindAttr, ::mlir::Type{}, "kind",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  initial_valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(initial_valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    initial_valueRawTypes[0] = type;
  }
  result.addTypes(sourceTypes);
  result.addTypes(initial_valueTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(initial_valueOperands, initial_valueTypes, initial_valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScanOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getKindAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getInitialValue();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"kind"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getInitialValue().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ScanOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ScanOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ScatterOp definitions
//===----------------------------------------------------------------------===//

ScatterOpAdaptor::ScatterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ScatterOpAdaptor::ScatterOpAdaptor(ScatterOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ScatterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScatterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ScatterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScatterOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScatterOpAdaptor::base() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ScatterOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange ScatterOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value ScatterOpAdaptor::getIndexVec() {
  return *getODSOperands(2).begin();
}

::mlir::Value ScatterOpAdaptor::index_vec() {
  return *getODSOperands(2).begin();
}

::mlir::Value ScatterOpAdaptor::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value ScatterOpAdaptor::mask() {
  return *getODSOperands(3).begin();
}

::mlir::Value ScatterOpAdaptor::getValueToStore() {
  return *getODSOperands(4).begin();
}

::mlir::Value ScatterOpAdaptor::valueToStore() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr ScatterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScatterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ScatterOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false, false, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 4) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ScatterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScatterOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value ScatterOp::base() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ScatterOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ScatterOp::indices() {
  return getODSOperands(1);
}

::mlir::Value ScatterOp::getIndexVec() {
  return *getODSOperands(2).begin();
}

::mlir::Value ScatterOp::index_vec() {
  return *getODSOperands(2).begin();
}

::mlir::Value ScatterOp::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value ScatterOp::mask() {
  return *getODSOperands(3).begin();
}

::mlir::Value ScatterOp::getValueToStore() {
  return *getODSOperands(4).begin();
}

::mlir::Value ScatterOp::valueToStore() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange ScatterOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getIndexVecMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::index_vecMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::getValueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ScatterOp::valueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScatterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScatterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value valueToStore) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(index_vec);
  odsState.addOperands(mask);
  odsState.addOperands(valueToStore);
}

void ScatterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::ValueRange indices, ::mlir::Value index_vec, ::mlir::Value mask, ::mlir::Value valueToStore) {
  odsState.addOperands(base);
  odsState.addOperands(indices);
  odsState.addOperands(index_vec);
  odsState.addOperands(mask);
  odsState.addOperands(valueToStore);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScatterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScatterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ScatterOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ScatterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::OpAsmParser::OperandType index_vecRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> index_vecOperands(index_vecRawOperands);  ::llvm::SMLoc index_vecOperandsLoc;
  (void)index_vecOperandsLoc;
  ::mlir::OpAsmParser::OperandType maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::OpAsmParser::OperandType valueToStoreRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueToStoreOperands(valueToStoreRawOperands);  ::llvm::SMLoc valueToStoreOperandsLoc;
  (void)valueToStoreOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type index_vecRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> index_vecTypes(index_vecRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type valueToStoreRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueToStoreTypes(valueToStoreRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  index_vecOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(index_vecRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valueToStoreOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueToStoreRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    index_vecRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueToStoreRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(index_vecOperands, index_vecTypes, index_vecOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueToStoreOperands, valueToStoreTypes, valueToStoreOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScatterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter << ' ' << "[";
  _odsPrinter << getIndexVec();
  _odsPrinter << "]";
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getValueToStore();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getIndexVec().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getValueToStore().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ScatterOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ScatterOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ShapeCastOp definitions
//===----------------------------------------------------------------------===//

ShapeCastOpAdaptor::ShapeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ShapeCastOpAdaptor::ShapeCastOpAdaptor(ShapeCastOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ShapeCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShapeCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShapeCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShapeCastOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShapeCastOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ShapeCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ShapeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShapeCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShapeCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShapeCastOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShapeCastOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ShapeCastOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShapeCastOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShapeCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShapeCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShapeCastOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value ShapeCastOp::result() {
  return *getODSResults(0).begin();
}

void ShapeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source) {
  odsState.addOperands(source);
  odsState.addTypes(result);
}

void ShapeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShapeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShapeCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShapeCastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ShapeCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType sourceRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  ::llvm::SMLoc sourceOperandsLoc;
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
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShapeCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ShapeCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ShapeCastOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::ShuffleOp definitions
//===----------------------------------------------------------------------===//

ShuffleOpAdaptor::ShuffleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ShuffleOpAdaptor::ShuffleOpAdaptor(ShuffleOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ShuffleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShuffleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShuffleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleOpAdaptor::getV1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleOpAdaptor::v1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleOpAdaptor::getV2() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShuffleOpAdaptor::v2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ShuffleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ShuffleOpAdaptor::getMaskAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("mask").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ShuffleOpAdaptor::getMask() {
  auto attr = getMaskAttr();
  return attr;
}

::mlir::ArrayAttr ShuffleOpAdaptor::maskAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("mask").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ShuffleOpAdaptor::mask() {
  auto attr = maskAttr();
  return attr;
}

::mlir::LogicalResult ShuffleOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_mask = odsAttrs.get("mask");
    if (!tblgen_mask)
      return emitError(loc, "'vector.shuffle' op ""requires attribute 'mask'");

    if (tblgen_mask && !(((tblgen_mask.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_mask.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.shuffle' op ""attribute 'mask' failed to satisfy constraint: 64-bit integer array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShuffleOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShuffleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleOp::getV1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleOp::v1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleOp::getV2() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShuffleOp::v2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ShuffleOp::getV1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleOp::v1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleOp::getV2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleOp::v2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShuffleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShuffleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleOp::getVector() {
  return *getODSResults(0).begin();
}

::mlir::Value ShuffleOp::vector() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ShuffleOp::getMaskAttr() {
  return (*this)->getAttr(getMaskAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ShuffleOp::getMask() {
  auto attr = getMaskAttr();
  return attr;
}

::mlir::ArrayAttr ShuffleOp::maskAttr() {
  return (*this)->getAttr(maskAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ShuffleOp::mask() {
  auto attr = maskAttr();
  return attr;
}

void ShuffleOp::setMaskAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMaskAttrName(), attr);
}

void ShuffleOp::maskAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(maskAttrName(), attr);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);
  odsState.addTypes(vector);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ShuffleOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShuffleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::mlir::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ShuffleOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ShuffleOp::verifyInvariantsImpl() {
  {
    auto tblgen_mask = (*this)->getAttr(getMaskAttrName());
    if (!tblgen_mask)
      return emitOpError("requires attribute 'mask'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_mask, "mask")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that first operand v1 and result have same element type");
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(1)))))
    return emitOpError("failed to verify that second operand v2 and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult ShuffleOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ShuffleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allOperands;
  ::mlir::ArrayAttr maskAttr;
  ::mlir::SmallVector<::mlir::Type, 1> allOperandTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(maskAttr, parser.getBuilder().getType<::mlir::NoneType>(), "mask",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(allOperandTypes))
    return ::mlir::failure();
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();

  ::llvm::SmallVector<::mlir::Type> inferredReturnTypes;
  if (::mlir::failed(ShuffleOp::inferReturnTypes(parser.getContext(),
      result.location, result.operands,
      result.attributes.getDictionary(parser.getContext()),
      result.regions, inferredReturnTypes)))
    return ::mlir::failure();
  result.addTypes(inferredReturnTypes);
  return ::mlir::success();
}

void ShuffleOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getMaskAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"mask"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperandTypes();
}

void ShuffleOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::ShuffleOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::SplatOp definitions
//===----------------------------------------------------------------------===//

SplatOpAdaptor::SplatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SplatOpAdaptor::SplatOpAdaptor(SplatOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SplatOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SplatOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SplatOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOpAdaptor::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value SplatOpAdaptor::input() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SplatOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SplatOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SplatOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SplatOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOp::getInput() {
  return *getODSOperands(0).begin();
}

::mlir::Value SplatOp::input() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SplatOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SplatOp::inputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SplatOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SplatOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplatOp::getAggregate() {
  return *getODSResults(0).begin();
}

::mlir::Value SplatOp::aggregate() {
  return *getODSResults(0).begin();
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value element, Type aggregateType) {
 build(odsBuilder, odsState, aggregateType, element); 
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type aggregate, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(aggregate);
}

void SplatOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplatOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SplatOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSResults(0).begin()).getType().cast<VectorType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that operand type matches element type of result");
  return ::mlir::success();
}

::mlir::LogicalResult SplatOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SplatOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type aggregateRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> aggregateTypes(aggregateRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    aggregateRawTypes[0] = type;
  }
  for (::mlir::Type type : aggregateTypes) {
    (void)type;
    if (!(((type.isa<::mlir::VectorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
      return parser.emitError(parser.getNameLoc()) << "'aggregate' must be vector of any type values, but got " << type;
    }
  }
  result.addTypes(aggregateTypes);
  if (parser.resolveOperands(inputOperands, aggregateTypes[0].cast<VectorType>().getElementType(), inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SplatOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getAggregate().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SplatOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::SplatOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::StoreOp definitions
//===----------------------------------------------------------------------===//

StoreOpAdaptor::StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

StoreOpAdaptor::StoreOpAdaptor(StoreOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange StoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOpAdaptor::getValueToStore() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOpAdaptor::valueToStore() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOpAdaptor::getBase() {
  return *getODSOperands(1).begin();
}

::mlir::Value StoreOpAdaptor::base() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange StoreOpAdaptor::getIndices() {
  return getODSOperands(2);
}

::mlir::ValueRange StoreOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr StoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
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

::mlir::Value StoreOp::getValueToStore() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOp::valueToStore() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOp::getBase() {
  return *getODSOperands(1).begin();
}

::mlir::Value StoreOp::base() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range StoreOp::getIndices() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range StoreOp::indices() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange StoreOp::getValueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::valueToStoreMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::baseMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::indicesMutable() {
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

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value valueToStore, ::mlir::Value base, ::mlir::ValueRange indices) {
  odsState.addOperands(valueToStore);
  odsState.addOperands(base);
  odsState.addOperands(indices);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value valueToStore, ::mlir::Value base, ::mlir::ValueRange indices) {
  odsState.addOperands(valueToStore);
  odsState.addOperands(base);
  odsState.addOperands(indices);
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
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType valueToStoreRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueToStoreOperands(valueToStoreRawOperands);  ::llvm::SMLoc valueToStoreOperandsLoc;
  (void)valueToStoreOperandsLoc;
  ::mlir::OpAsmParser::OperandType baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type valueToStoreRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueToStoreTypes(valueToStoreRawTypes);

  valueToStoreOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueToStoreRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
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
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueToStoreRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(valueToStoreOperands, valueToStoreTypes, valueToStoreOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValueToStore();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getValueToStore().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void StoreOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::StoreOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TransferReadOp definitions
//===----------------------------------------------------------------------===//

TransferReadOpAdaptor::TransferReadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TransferReadOpAdaptor::TransferReadOpAdaptor(TransferReadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange TransferReadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TransferReadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange TransferReadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransferReadOpAdaptor::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransferReadOpAdaptor::source() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange TransferReadOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::ValueRange TransferReadOpAdaptor::indices() {
  return getODSOperands(1);
}

::mlir::Value TransferReadOpAdaptor::getPadding() {
  return *getODSOperands(2).begin();
}

::mlir::Value TransferReadOpAdaptor::padding() {
  return *getODSOperands(2).begin();
}

::mlir::Value TransferReadOpAdaptor::getMask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TransferReadOpAdaptor::mask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr TransferReadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr TransferReadOpAdaptor::getPermutationMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("permutation_map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap TransferReadOpAdaptor::getPermutationMap() {
  auto attr = getPermutationMapAttr();
  return attr.getValue();
}

::mlir::AffineMapAttr TransferReadOpAdaptor::permutation_mapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("permutation_map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap TransferReadOpAdaptor::permutation_map() {
  auto attr = permutation_mapAttr();
  return attr.getValue();
}

::mlir::ArrayAttr TransferReadOpAdaptor::getInBoundsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("in_bounds").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TransferReadOpAdaptor::getInBounds() {
  auto attr = getInBoundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr TransferReadOpAdaptor::in_boundsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("in_bounds").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TransferReadOpAdaptor::in_bounds() {
  auto attr = in_boundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult TransferReadOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'vector.transfer_read' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'vector.transfer_read' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_permutation_map = odsAttrs.get("permutation_map");
    if (!tblgen_permutation_map)
      return emitError(loc, "'vector.transfer_read' op ""requires attribute 'permutation_map'");

    if (tblgen_permutation_map && !((tblgen_permutation_map.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'vector.transfer_read' op ""attribute 'permutation_map' failed to satisfy constraint: AffineMap attribute");
  }
  {
    auto tblgen_in_bounds = odsAttrs.get("in_bounds");
    if (tblgen_in_bounds && !(((tblgen_in_bounds.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_in_bounds.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::BoolAttr>()); }))))
      return emitError(loc, "'vector.transfer_read' op ""attribute 'in_bounds' failed to satisfy constraint: 1-bit boolean array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransferReadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range TransferReadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransferReadOp::getSource() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransferReadOp::source() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range TransferReadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range TransferReadOp::indices() {
  return getODSOperands(1);
}

::mlir::Value TransferReadOp::getPadding() {
  return *getODSOperands(2).begin();
}

::mlir::Value TransferReadOp::padding() {
  return *getODSOperands(2).begin();
}

::mlir::Value TransferReadOp::getMask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TransferReadOp::mask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange TransferReadOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::getPaddingMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::paddingMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferReadOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TransferReadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransferReadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransferReadOp::getVector() {
  return *getODSResults(0).begin();
}

::mlir::Value TransferReadOp::vector() {
  return *getODSResults(0).begin();
}

::mlir::AffineMapAttr TransferReadOp::getPermutationMapAttr() {
  return (*this)->getAttr(getPermutationMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap TransferReadOp::getPermutationMap() {
  auto attr = getPermutationMapAttr();
  return attr.getValue();
}

::mlir::AffineMapAttr TransferReadOp::permutation_mapAttr() {
  return (*this)->getAttr(permutation_mapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap TransferReadOp::permutation_map() {
  auto attr = permutation_mapAttr();
  return attr.getValue();
}

::mlir::ArrayAttr TransferReadOp::getInBoundsAttr() {
  return (*this)->getAttr(getInBoundsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TransferReadOp::getInBounds() {
  auto attr = getInBoundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr TransferReadOp::in_boundsAttr() {
  return (*this)->getAttr(in_boundsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TransferReadOp::in_bounds() {
  auto attr = in_boundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void TransferReadOp::setPermutationMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getPermutationMapAttrName(), attr);
}

void TransferReadOp::permutation_mapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(permutation_mapAttrName(), attr);
}

void TransferReadOp::setInBoundsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInBoundsAttrName(), attr);
}

void TransferReadOp::in_boundsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(in_boundsAttrName(), attr);
}

::mlir::Attribute TransferReadOp::removeIn_boundsAttr() {
  return (*this)->removeAttr(getInBoundsAttrName());
}

void TransferReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addOperands(padding);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), 1, (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), permutation_map);
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  odsState.addTypes(vector);
}

void TransferReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addOperands(padding);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), 1, (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), permutation_map);
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransferReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addOperands(padding);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), 1, (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(permutation_map));
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  odsState.addTypes(vector);
}

void TransferReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, ::mlir::Value padding, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(source);
  odsState.addOperands(indices);
  odsState.addOperands(padding);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(indices.size()), 1, (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(permutation_map));
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransferReadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransferReadOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(getOperandSegmentSizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_permutation_map = (*this)->getAttr(getPermutationMapAttrName());
    if (!tblgen_permutation_map)
      return emitOpError("requires attribute 'permutation_map'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps7(*this, tblgen_permutation_map, "permutation_map")))
      return ::mlir::failure();
  }
  {
    auto tblgen_in_bounds = (*this)->getAttr(getInBoundsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps8(*this, tblgen_in_bounds, "in_bounds")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransferReadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::TransferReadOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TransferWriteOp definitions
//===----------------------------------------------------------------------===//

TransferWriteOpAdaptor::TransferWriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TransferWriteOpAdaptor::TransferWriteOpAdaptor(TransferWriteOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange TransferWriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TransferWriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange TransferWriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransferWriteOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransferWriteOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransferWriteOpAdaptor::getSource() {
  return *getODSOperands(1).begin();
}

::mlir::Value TransferWriteOpAdaptor::source() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange TransferWriteOpAdaptor::getIndices() {
  return getODSOperands(2);
}

::mlir::ValueRange TransferWriteOpAdaptor::indices() {
  return getODSOperands(2);
}

::mlir::Value TransferWriteOpAdaptor::getMask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TransferWriteOpAdaptor::mask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr TransferWriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::AffineMapAttr TransferWriteOpAdaptor::getPermutationMapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("permutation_map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap TransferWriteOpAdaptor::getPermutationMap() {
  auto attr = getPermutationMapAttr();
  return attr.getValue();
}

::mlir::AffineMapAttr TransferWriteOpAdaptor::permutation_mapAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::AffineMapAttr attr = odsAttrs.get("permutation_map").cast<::mlir::AffineMapAttr>();
  return attr;
}

::mlir::AffineMap TransferWriteOpAdaptor::permutation_map() {
  auto attr = permutation_mapAttr();
  return attr.getValue();
}

::mlir::ArrayAttr TransferWriteOpAdaptor::getInBoundsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("in_bounds").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TransferWriteOpAdaptor::getInBounds() {
  auto attr = getInBoundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr TransferWriteOpAdaptor::in_boundsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("in_bounds").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TransferWriteOpAdaptor::in_bounds() {
  auto attr = in_boundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult TransferWriteOpAdaptor::verify(::mlir::Location loc) {
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'vector.transfer_write' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'vector.transfer_write' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_permutation_map = odsAttrs.get("permutation_map");
    if (!tblgen_permutation_map)
      return emitError(loc, "'vector.transfer_write' op ""requires attribute 'permutation_map'");

    if (tblgen_permutation_map && !((tblgen_permutation_map.isa<::mlir::AffineMapAttr>())))
      return emitError(loc, "'vector.transfer_write' op ""attribute 'permutation_map' failed to satisfy constraint: AffineMap attribute");
  }
  {
    auto tblgen_in_bounds = odsAttrs.get("in_bounds");
    if (tblgen_in_bounds && !(((tblgen_in_bounds.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_in_bounds.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::BoolAttr>()); }))))
      return emitError(loc, "'vector.transfer_write' op ""attribute 'in_bounds' failed to satisfy constraint: 1-bit boolean array attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransferWriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range TransferWriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransferWriteOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransferWriteOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransferWriteOp::getSource() {
  return *getODSOperands(1).begin();
}

::mlir::Value TransferWriteOp::source() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range TransferWriteOp::getIndices() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range TransferWriteOp::indices() {
  return getODSOperands(2);
}

::mlir::Value TransferWriteOp::getMask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TransferWriteOp::mask() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange TransferWriteOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::vectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::getSourceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::sourceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::indicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TransferWriteOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TransferWriteOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range TransferWriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransferWriteOp::getResult() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::Value TransferWriteOp::result() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::AffineMapAttr TransferWriteOp::getPermutationMapAttr() {
  return (*this)->getAttr(getPermutationMapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap TransferWriteOp::getPermutationMap() {
  auto attr = getPermutationMapAttr();
  return attr.getValue();
}

::mlir::AffineMapAttr TransferWriteOp::permutation_mapAttr() {
  return (*this)->getAttr(permutation_mapAttrName()).cast<::mlir::AffineMapAttr>();
}

::mlir::AffineMap TransferWriteOp::permutation_map() {
  auto attr = permutation_mapAttr();
  return attr.getValue();
}

::mlir::ArrayAttr TransferWriteOp::getInBoundsAttr() {
  return (*this)->getAttr(getInBoundsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TransferWriteOp::getInBounds() {
  auto attr = getInBoundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr TransferWriteOp::in_boundsAttr() {
  return (*this)->getAttr(in_boundsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TransferWriteOp::in_bounds() {
  auto attr = in_boundsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void TransferWriteOp::setPermutationMapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(getPermutationMapAttrName(), attr);
}

void TransferWriteOp::permutation_mapAttr(::mlir::AffineMapAttr attr) {
  (*this)->setAttr(permutation_mapAttrName(), attr);
}

void TransferWriteOp::setInBoundsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInBoundsAttrName(), attr);
}

void TransferWriteOp::in_boundsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(in_boundsAttrName(), attr);
}

::mlir::Attribute TransferWriteOp::removeIn_boundsAttr() {
  return (*this)->removeAttr(getInBoundsAttrName());
}

void TransferWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(vector);
  odsState.addOperands(source);
  odsState.addOperands(indices);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), permutation_map);
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  if (result)
    odsState.addTypes(result);
}

void TransferWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMapAttr permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(vector);
  odsState.addOperands(source);
  odsState.addOperands(indices);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), permutation_map);
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  odsState.addTypes(resultTypes);
}

void TransferWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type result, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(vector);
  odsState.addOperands(source);
  odsState.addOperands(indices);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(permutation_map));
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  if (result)
    odsState.addTypes(result);
}

void TransferWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices, ::mlir::AffineMap permutation_map, /*optional*/::mlir::Value mask, /*optional*/::mlir::ArrayAttr in_bounds) {
  odsState.addOperands(vector);
  odsState.addOperands(source);
  odsState.addOperands(indices);
  if (mask)
    odsState.addOperands(mask);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, 1, static_cast<int32_t>(indices.size()), (mask ? 1 : 0)}));
  odsState.addAttribute(getPermutationMapAttrName(odsState.name), ::mlir::AffineMapAttr::get(permutation_map));
  if (in_bounds) {
  odsState.addAttribute(getInBoundsAttrName(odsState.name), in_bounds);
  }
  odsState.addTypes(resultTypes);
}

void TransferWriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransferWriteOp::verifyInvariantsImpl() {
  {
    auto sizeAttr = (*this)->getAttr(getOperandSegmentSizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_permutation_map = (*this)->getAttr(getPermutationMapAttrName());
    if (!tblgen_permutation_map)
      return emitOpError("requires attribute 'permutation_map'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps7(*this, tblgen_permutation_map, "permutation_map")))
      return ::mlir::failure();
  }
  {
    auto tblgen_in_bounds = (*this)->getAttr(getInBoundsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps8(*this, tblgen_in_bounds, "in_bounds")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransferWriteOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::TransferWriteOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TransposeOp definitions
//===----------------------------------------------------------------------===//

TransposeOpAdaptor::TransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange TransposeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TransposeOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeOpAdaptor::vector() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TransposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TransposeOpAdaptor::getTranspAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("transp").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeOpAdaptor::getTransp() {
  auto attr = getTranspAttr();
  return attr;
}

::mlir::ArrayAttr TransposeOpAdaptor::transpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::ArrayAttr attr = odsAttrs.get("transp").cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TransposeOpAdaptor::transp() {
  auto attr = transpAttr();
  return attr;
}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_transp = odsAttrs.get("transp");
    if (!tblgen_transp)
      return emitError(loc, "'vector.transpose' op ""requires attribute 'transp'");

    if (tblgen_transp && !(((tblgen_transp.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_transp.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return ((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))); }))))
      return emitError(loc, "'vector.transpose' op ""attribute 'transp' failed to satisfy constraint: 64-bit integer array attribute");
  }
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

::mlir::Value TransposeOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value TransposeOp::vector() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TransposeOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TransposeOp::vectorMutable() {
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

::mlir::Value TransposeOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value TransposeOp::result() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr TransposeOp::getTranspAttr() {
  return (*this)->getAttr(getTranspAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeOp::getTransp() {
  auto attr = getTranspAttr();
  return attr;
}

::mlir::ArrayAttr TransposeOp::transpAttr() {
  return (*this)->getAttr(transpAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TransposeOp::transp() {
  auto attr = transpAttr();
  return attr;
}

void TransposeOp::setTranspAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTranspAttrName(), attr);
}

void TransposeOp::transpAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(transpAttrName(), attr);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value vector, ::mlir::ArrayAttr transp) {
  odsState.addOperands(vector);
  odsState.addAttribute(getTranspAttrName(odsState.name), transp);
  odsState.addTypes(result);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::ArrayAttr transp) {
  odsState.addOperands(vector);
  odsState.addAttribute(getTranspAttrName(odsState.name), transp);
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
  {
    auto tblgen_transp = (*this)->getAttr(getTranspAttrName());
    if (!tblgen_transp)
      return emitOpError("requires attribute 'transp'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_VectorOps0(*this, tblgen_transp, "transp")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((getElementTypeOrSelf((*this->getOperation()).getResult(0)) == getElementTypeOrSelf((*this->getOperation()).getOperand(0)))))
    return emitOpError("failed to verify that operand and result have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult TransposeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType vectorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vectorOperands(vectorRawOperands);  ::llvm::SMLoc vectorOperandsLoc;
  (void)vectorOperandsLoc;
  ::mlir::ArrayAttr transpAttr;
  ::mlir::Type vectorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> vectorTypes(vectorRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  vectorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vectorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(transpAttr, parser.getBuilder().getType<::mlir::NoneType>(), "transp",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    vectorRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::VectorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(vectorOperands, vectorTypes, vectorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TransposeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVector();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTranspAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"transp"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVector().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::VectorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TransposeOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::TransposeOp)

namespace mlir {
namespace vector {

//===----------------------------------------------------------------------===//
// ::mlir::vector::TypeCastOp definitions
//===----------------------------------------------------------------------===//

TypeCastOpAdaptor::TypeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TypeCastOpAdaptor::TypeCastOpAdaptor(TypeCastOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange TypeCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TypeCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TypeCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TypeCastOpAdaptor::getMemref() {
  return *getODSOperands(0).begin();
}

::mlir::Value TypeCastOpAdaptor::memref() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TypeCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TypeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TypeCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TypeCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TypeCastOp::getMemref() {
  return *getODSOperands(0).begin();
}

::mlir::Value TypeCastOp::memref() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TypeCastOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TypeCastOp::memrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TypeCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TypeCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TypeCastOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value TypeCastOp::result() {
  return *getODSResults(0).begin();
}

void TypeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref) {
  odsState.addOperands(memref);
  odsState.addTypes(result);
}

void TypeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TypeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TypeCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_VectorOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TypeCastOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TypeCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::OperandType memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
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
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TypeCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
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

void TypeCastOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace vector
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::vector::TypeCastOp)


#endif  // GET_OP_CLASSES

